#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <random>
#include <chrono>
#include <fstream>
#include <sstream>
#include <utility>
#include <curand.h>
#include <curand_kernel.h>

using namespace std;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }  // Standart Error Managemant mit CUDA
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void setup_kernel(curandState *state, int N, unsigned long long seed){

    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    for (int j = idx; j<N; j += blockDim.x * gridDim.x) {
        curand_init(seed, j, 0, &state[j]);                   // inizialisiert in jedem Thread einen eigenen Zufallszahlengenerator
    }
}

__global__ void tour_konstruktions_kernel(
    curandState *my_curandstate, 
    int N,          // Anzahl der Ameisen insgesamt
    int cldim,      // größe der TSP-Instanz
    float alpha,
    float beta,
    int *cost, 
    float *phero,
    int *d_route    // array für Routenrückgabe
    ) {

    extern __shared__ int s[];// Shared variables inizialisierung
    int* route = s;
    float* probabilities = (float*)&route[cldim*blockDim.x];
    bool* visited = (bool*)&probabilities[cldim*blockDim.x];

    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    int tid = threadIdx.x;
  
    for (int j = idx; j<N; j += blockDim.x * gridDim.x) { // jeder Thread berechnet eine ganze Tour
        for (int i = 0; i < cldim; i++) visited[tid*cldim+i] = false;
        float myrandstart = curand_uniform(my_curandstate+idx); // Startstadt bestimmen
        myrandstart *= (cldim -1 +0.99999);
        int start = (int)truncf(myrandstart);
        route[tid*cldim] = start;
        visited[tid*cldim+start] = true;
        probabilities[tid*cldim+start] = 0;
        for (int i = 1; i < cldim-1; i++) { //alle anderen Städte bis auf die letzte bestimmen.
            int current = route[tid*cldim+i-1];
            float sum = 0;
            for (int j = 0; j < cldim; j++) {//Wahrscheinlcihkeiten berechenn und gleich aufsummieren
                if (!visited[tid*cldim+j]) { 
                    probabilities[tid*cldim+j] = __powf(phero[current*cldim+j]+0.1E-28, alpha) * __powf(1./(cost[current*cldim+j]+0.1E-3), beta);
                    sum += probabilities[tid*cldim+j];
                }
            }
            for (int j = 0; j < cldim; j++) {//Wahrscheinlcihkeiten durch Summe teilen
                if (!visited[tid*cldim+j]) {
                    probabilities[tid*cldim+j] /= sum;
                }
            }
            float r = curand_uniform(my_curandstate+idx);     // r entspricht einem Wert aus der Menge (0,1]
            float sum_prob = 0;
            int next = -1;
            for (int j = 0; j < cldim; j++) {                   // roulett wheel selection
                sum_prob += probabilities[tid*cldim+j]; 
                if (r <= sum_prob) {
                    next = j;
                    break;
                }
            }
            if (next == -1 && r > sum_prob) {       // prüft, ob das Problem die ungenauigkeit der Errechneten Wahrscheinlichkeiten ist.
                int j = cldim-1;
                while (visited[tid*cldim+j]) j--;
                next = j;
            }
            route[tid*cldim+i] = next;
            visited[tid*cldim+next] = true;             
            probabilities[tid*cldim+next] = 0;
        }
        int i = 0;
        while (visited[tid*cldim+i]) i++;                   // wählt die letzte verbleibende Stadt nach Ausschlussverfahren.
        route[tid*cldim+cldim-1] = i;
        for (int i = 0; i < cldim; i++) d_route[idx*cldim+i] = route[tid*cldim+i];
    }
}

__global__ void pheromon_aktualisierungs_kernel( // seriell, wird nur von einem Thread in einem einzigen Block ausgeführt.
    float p,
    int N, 
    int cldim,
    int lenofbestwaysofar,
    int *bestroute,
    int *cost, 
    float *phero,
    int *route
    ) {

    for (int i = 0; i < cldim*cldim-1; i++) { //evaporation
        phero[i] = phero[i] * (1-p);
    }

    for (int antid = 0; antid < N; antid++) {  
        int len = 0;
        for (int i = 0; i < cldim-1; i++) {
            len += cost[route[antid*cldim + i]*cldim+route[antid*cldim + i+1]]; //berechnung der Tourlänge
        }
        len += cost[route[antid*cldim + cldim-1]*cldim+route[antid*cldim]];
        
        if (len < lenofbestwaysofar) { // überprüfen ob eine bessere Tour gefundne wurde
            for (int i = 0; i < cldim; i++) {
                bestroute[i] = route[antid*cldim + i];
            }
            lenofbestwaysofar = len;
        }

        float nlen = 1./len;

        for (int i = 0; i < cldim-1; i++) {// HInzufügend er Pheromone
            phero[route[antid*cldim + i]*cldim+route[antid*cldim + i+1]] += nlen;
            phero[route[antid*cldim + i+1]*cldim+route[antid*cldim + i]] += nlen;
        }
        phero[route[antid*cldim + cldim-1]*cldim+route[antid*cldim]] += nlen;
        phero[route[antid*cldim]*cldim+route[antid*cldim + cldim-1]] += nlen;   
    }
}

vector<pair<float, float>> parseTSPFile(const string& filenameshort) {// utility
    vector<pair<float, float>> coordinates;

    string filename = "C:/Users/jonie/Desktop/hauptseminarpp/hausarbeit/testtsps/" + filenameshort +".tsp";
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        return coordinates;
    }

    string line;
    // Skip header lines
    bool sectionFound = false;
    while (getline(file, line)) {
        if (line == "NODE_COORD_SECTION") {
            sectionFound = true;
            break;
        }
    }

    if (!sectionFound) {
        cerr << "NODE_COORD_SECTION not found in file: " << filename << endl;
        return coordinates;
    }

    // Read node coordinates
    int index;
    float x, y;
    while (file >> index >> x >> y) {
        coordinates.push_back(make_pair(x, y));
    }

    file.close();
    return coordinates;
}

class ac {
    private:
        int *cost;    //ohne d CPU
        int *d_cost;  // mit d device/GPU
        float *phero;
        float *d_phero;
        int *d_route;
        int *d_bestwaysofar;
        vector<pair<float, float>> cl;
        int cldim;
        float alpha;
        float beta;
        int *bestwaysofar;
        vector<int> vbestwaysofar;
        int lenofbestwaysofar;
        int lenofbestway;
        bool solisopt;
        random_device rd;
        mt19937 gen;
        curandState *d_state;
        int N;
        int block_size;
        int blocks;
        unsigned long long seed;
        void inizialisiereACO(vector<pair<float, float>> citylist, float lenofbestroute, int anzAnts) {
            cl = citylist;
            cldim = citylist.size();
            alpha = 1;      //festlegung von alpha und beta
            beta = 2;
            lenofbestwaysofar = 0;              //
            lenofbestway = lenofbestroute;      // wird nur benötigt wenn eine optimale Route gefunden werden soll.
            solisopt = false;                   //
            N = anzAnts;
            seed = time(NULL); // seed für random generator 
            // malloc und inizialisierung
            cost = (int *)malloc((cldim*cldim)*sizeof(int));
            for (int x = 0; x < cldim; x++) {
                for (int y = x+1; y < cldim; y++) {
                    float way = round(sqrt(pow(cl[x].first - cl[y].first, 2) + pow(cl[x].second - cl[y].second, 2)));
                    cost[x*cldim+y] = way;
                    cost[y*cldim+x] = way;
                }
            }
            for (int x = 0; x < cldim; x++) cost[x*cldim+x] = 0;
            phero = (float *)malloc((cldim*cldim)*sizeof(float));
            for (int x = 0; x < cldim*cldim; x++) phero[x] = 0;
            bestwaysofar = (int *)malloc(cldim*sizeof(int));
            for (int i = 0; i < cldim; i++) {
                bestwaysofar[i] = i;
                vbestwaysofar.push_back(i);
            }
            lenofbestwaysofar = calulate_way_from_route(vbestwaysofar);
        }
        void initialisiereGPU() { // cudamalloc, cudaCopy und setupkernel 
            gpuErrchk(cudaMalloc(&d_state, N*sizeof(curandState)));
            block_size = 4; // threads pro Block hier verrändern
            blocks = (N / block_size) + (N % block_size == 0 ? 0:1); // alle hier benutzten Werte sind durch 2er Potenzen teilbar. Trotzdem hier der Sicherheitsmechanismus.
            setup_kernel<<<blocks,block_size>>>(d_state, N, seed);

            gpuErrchk(cudaMalloc((void **) &d_cost, cldim*cldim*sizeof(int)));
            gpuErrchk(cudaMemcpy(d_cost, cost, cldim*cldim*sizeof(int), cudaMemcpyHostToDevice));
            gpuErrchk(cudaMalloc((void **) &d_phero, cldim*cldim*sizeof(float)));
            gpuErrchk(cudaMemcpy(d_phero, phero, cldim*cldim*sizeof(float), cudaMemcpyHostToDevice));
            gpuErrchk(cudaMalloc((void **) &d_bestwaysofar, cldim*sizeof(int)));
            gpuErrchk(cudaMemcpy(d_bestwaysofar, bestwaysofar, cldim*sizeof(int), cudaMemcpyHostToDevice));
            gpuErrchk(cudaMalloc((void **) &d_route, N*cldim*sizeof(int)));
        }
        void oneIteration(float p) {
            //kernel call
            tour_konstruktions_kernel<<<blocks, block_size, cldim*block_size*sizeof(int)+cldim*block_size*sizeof(float)+cldim*block_size*sizeof(bool)>>>(d_state, N, cldim, alpha, beta, d_cost, d_phero, d_route);
            gpuErrchk(cudaPeekAtLastError());
            gpuErrchk(cudaDeviceSynchronize());
            pheromon_aktualisierungs_kernel<<<1,1>>>(p, N, cldim, lenofbestwaysofar, d_bestwaysofar, d_cost, d_phero, d_route);
            gpuErrchk(cudaPeekAtLastError());   
            gpuErrchk(cudaMemcpy(bestwaysofar, d_bestwaysofar, cldim*sizeof(int), cudaMemcpyDeviceToHost));
            //aktualisierung der besten bisher gefundenen Tour
            vbestwaysofar.clear();  
            for (int i = 0; i < cldim; i++) {
                vbestwaysofar.push_back(bestwaysofar[i]);
            }
            lenofbestwaysofar = calulate_way_from_route(vbestwaysofar);
            if (lenofbestwaysofar <= lenofbestway) {
                solisopt = true; 
            }
        }
        int calulate_way_from_route(vector<int> route) { //utility
            int way = 0;
            for (int i = 0; i < cldim-1; i++) {
                way += cost[route[i]*cldim+route[i+1]];
            }
            way += cost[route[cldim-1]*cldim+route[0]];
            return way;
        }
    public:
        ac(vector<pair<float, float>> citylist, int lenofbestroute=0, int anzAnts=2048) : gen(rd()) {
            inizialisiereACO(citylist, lenofbestroute, anzAnts);
            initialisiereGPU();
        }
        void doIteration(float p=0.5) {
            oneIteration(p);
            if (solisopt) return;
        }
        vector<vector<int>> getcost() {
            vector<vector<int>> resultmatrix (cldim, vector<int>(cldim));
            for (int i = 0; i < cldim; i++) {
                for (int j = 0; j < cldim; j++) {
                    resultmatrix[i][j] = cost[i * cldim + j];
                }
            }
            return resultmatrix;
        }
        vector<vector<float>> getphero() {
            vector<vector<float>> resultmatrix (cldim, vector<float>(cldim));
            for (int i = 0; i < cldim; i++) {
                for (int j = 0; j < cldim; j++) {
                    resultmatrix[i][j] = phero[i * cldim + j];
                }
            }
            return resultmatrix;
        }
        vector<int> getbestroute() {
            return vbestwaysofar;
        }
        int getbestroutelen() {
            return lenofbestwaysofar;
        }
        bool issolopt() {
            return solisopt;
        }
        void freeall() {
            free(cost);
            free(phero);
            free(bestwaysofar);
            cudaFree(d_state);
            cudaFree(d_cost);
            cudaFree(d_phero);
            cudaFree(d_route);
            cudaFree(d_bestwaysofar);
        }
};

int main(void) {
    
    vector<pair<float, float>> dj38 = parseTSPFile("dj38");
    int soldj38 = 6656;

    vector<pair<float, float>> lu980 = parseTSPFile("lu980");
    int sollu980 = 11340;

    vector<pair<float, float>> qa194 = parseTSPFile("qa194");
    int solqa194 = 9352;

    vector<pair<float, float>> a280 = parseTSPFile("a280");
    int sola280 = 2579;
    
    vector<pair<float, float>> d198 = parseTSPFile("d198");
    int sold198 = 15780;

    vector<pair<float, float>> lin318 = parseTSPFile("lin318");
    int sollin318 = 42029;

    vector<pair<float, float>> pcb442 = parseTSPFile("pcb442");
    int solpcb442 = 50778;

    vector<pair<float, float>> pr1002 = parseTSPFile("pr1002");
    int solpr1002 = 259045;

    vector<pair<float, float>> rat783 = parseTSPFile("rat783");
    int solrat783 = 8806;

    vector<int> coloniesize = {1024, 2048, 4096, 8192};

/*
    // Berechnung der Durchschnittslänge
    for (int i = 0; i < coloniesize.size(); i++) {

        int anzberechungen = 30;
        vector<int> mlength;
        mlength.resize(anzberechungen);

        for (int k = 0; k < anzberechungen; k++) { 

            int anziter = 30;
            vector<pair<float, float>> citylits = dj38;
            float p = 0.5;

            int bestroutlen = INT_MAX;
            ac region(citylits, 0, coloniesize[i]);

            for (int j = 0; j < anziter; j++) {
                region.doIteration(p);      
            }
            bestroutlen = region.getbestroutelen(); 
            //cout << "bestroutelen: " << bestroutlen << endl;
            mlength[k] = bestroutlen;
            region.freeall();
      
        }

        float summe = 0.0;
        for (int j = 0; j < anzberechungen; j++) {
            summe += mlength[j];
        }
        float avg = summe / mlength.size();
        cout << "Die Durchschnittliche routenlaenge fuer " << coloniesize[i] << " betraegt: " << avg << endl;
    }
*/

    // Berechnung der Durchschnittszeit
    for (int i = 0; i < coloniesize.size(); i++) {

        int anzberechungen = 30;
        vector<pair<float, float>> citylits = qa194;
        float p = 0.5;
        vector<chrono::duration<float>> listofdurations;
        listofdurations.resize(anzberechungen);

        vector<int> bestrout;
        int bestroutlen = INT_MAX;
        ac region(citylits, 0, coloniesize[i]);

        for (int j = 0; j < anzberechungen; j++) {
            auto start = chrono::high_resolution_clock::now();

            region.doIteration(p);

            auto end = chrono::high_resolution_clock::now();
            listofdurations[j] = end - start;

            bestroutlen = region.getbestroutelen();
            bestrout = region.getbestroute();
            
        }
        
        cout << "bestroutelen: " << bestroutlen << endl;
        bestrout = region.getbestroute();
        cout << "bestroute: [";
        for (const auto& element : bestrout) {
            cout << element << ", ";
        }
        cout << endl;
        region.freeall();

        float summe = 0.0;
        for (int j = 0; j < anzberechungen; j++) {
            summe += listofdurations[j].count();
        }

        float avg = summe / listofdurations.size();
        cout << "Die Durchschnittliche Ausfuehrungszeit fuer " << coloniesize[i] << " betraegt: " << avg << " Sekunden." << endl;
        
        cout << "suration values: ";
        for (const auto& element : listofdurations) {
            cout << element.count() << " ";
        }
        cout << endl;
        
    }

    return 0;
} 