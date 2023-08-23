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

__global__ void setup_kernel(curandState *state, int N, unsigned long long seed){

    int ameisenid = blockIdx.x;
    if (threadIdx.x == 0) {
        curand_init(seed, ameisenid, 0, &state[ameisenid]);     // inizialisiert in jedem Block einen eigenen Zufallszahlengenerator
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

    extern __shared__ int s[];   // Shared variables inizialisierung
    int* route = s;
    float* probabilities = (float*)&route[cldim];
    bool* visited = (bool*)&probabilities[cldim];
    __shared__ float sum;

    int ameisenid = blockIdx.x; 
    int stadtid = threadIdx.x;

    if (stadtid < cldim) {
        if (stadtid == 0) {
            for (int i = 0; i < cldim; i++) visited[i] = false;  // visited wieder mit false inizialisieren
            float myrandstart = curand_uniform(my_curandstate+ameisenid); // bestimmen der Ausgangsstadt
            myrandstart *= (cldim -1 +0.99999); // its very important that this is not even one 9 longer, since above .999999 the Computer would roud up, leading to an Error
            int start = (int)truncf(myrandstart);
            route[0] = start;
            visited[start] = true;
        }
        __syncthreads();
        for (int i = 1; i < cldim-1; i++) { //alle anderen Städte bis auf die letzte bestimmen.
            int aktuellestadt = route[i-1];
            if (!visited[stadtid]) { //Wahrscheinlichkietsberechnung nur bei sädten die noch nicht besucht wurden
                probabilities[stadtid] = __powf(phero[aktuellestadt*cldim+stadtid]+0.1E-28, alpha) * __powf(1./(cost[aktuellestadt*cldim+stadtid]+0.1E-3), beta);
            } else {
                probabilities[stadtid] = 0;
            }
            __syncthreads();
            if (stadtid == 0) {
                sum = 0;
                for (int j = 0; j < cldim; j++) {
                    sum += probabilities[j];        // summe bilden
                }
            }
            __syncthreads();
            probabilities[stadtid] /= sum;  // durch summe Teilen
            __syncthreads();
            if (stadtid == 0) {    // roulett wheel selection
                float r = curand_uniform(my_curandstate+ameisenid); // r in Menge (0,1]
                float sum_prob = 0;
                int next = -1;
                for (int j = 0; j < cldim; j++) {
                    sum_prob += probabilities[j];
                    if (r <= sum_prob) {
                        next = j;
                        break;
                    }
                }
                if (next == -1 && r > sum_prob) {       // prüft, ob das Problem die ungenauigkeit der Errechneten Wahrscheinlichkeiten ist.
                    int j = cldim-1;
                    while (visited[j]) j--;
                    next = j;                       // setzen auf deie letzte nciht besuchte Stadt
                }
                route[i] = next;
                visited[next] = true;
            }
            __syncthreads();
        }

        if (stadtid == 0) {
            int i = 0;
            while (visited[i]) i++;
            route[cldim-1] = i;         // letzte Stadt per Ausschluss bestimmen
            for (int i = 0; i < cldim; i++) d_route[ameisenid*cldim+i] = route[i]; // Route übertragen
        }
    }
}

__global__ void pheromon_evaporation_kernel(
    float p,
    float *phero
    ) {

    int kante = blockIdx.x * blockDim.x + threadIdx.x; 
    phero[kante] = phero[kante] * (1-p);

}

__global__ void pheromon_aktualisierungs_kernel( 
    int cldim,
    int *lenlist,
    int *cost, 
    float *phero,
    int *route
    ) {
    extern __shared__ int partial_len[];
    __shared__ float pherodelta;

    int ameisenid = blockIdx.x;
    int schrittid = threadIdx.x;
    int stadt_i, stadt_j; 
    if (schrittid == 0) { // die einzellnen Threads suchen nur den weg von einem zum anderen Knoten heraus 
        stadt_i = route[ameisenid*cldim + cldim - 1];
        stadt_j = route[ameisenid*cldim];
    } else {
        stadt_i = route[ameisenid*cldim + schrittid - 1];
        stadt_j = route[ameisenid*cldim + schrittid];
    }

    partial_len[schrittid] = cost[stadt_i * cldim + stadt_j];
    __syncthreads();
    if (schrittid == 0) { // ein einzelner Thread im Block addiet es zur Tourlaenge zusammen und gibt den wert zurück
        int len = 0;
        for (int i = 0; i < cldim; i++) len += partial_len[i];
        //write it down, analyse it later sequentialy.
        lenlist[ameisenid] = len; 
        pherodelta = 1./len;
    }
    __syncthreads();
    atomicAdd(&phero[stadt_i * cldim + stadt_j], pherodelta); // hinzufügen der Pheromone
    atomicAdd(&phero[stadt_j * cldim + stadt_i], pherodelta);
}

vector<pair<float, float>> parseTSPFile(const string& filenameshort) {
    vector<pair<float, float>> coordinates;

    string filename = "C:/Users/jonie/Desktop/hausarbeit/testtsps/" + filenameshort +".tsp"; //hier lokalen weg zu den tsp instanzen einfügen.
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
        int *cost;     // Kosten im CPU
        int *d_cost;   // Kosten im GPU
        float *phero;
        float *d_phero;
        int *d_route;
        int *route;
        int *d_lenlist;
        int *lenlist;
        vector<pair<float, float>> cl;
        int cldim;
        float alpha;
        float beta;
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
        void inizialisiereACO(vector<pair<float, float>> citylist, int lenofbestroute, int anzAnts) {
            cl = citylist;
            cldim = citylist.size();
            alpha = 1;          // festlegung der werte alpha und Beta
            beta = 2;
            lenofbestwaysofar = 0;              //
            lenofbestway = lenofbestroute;      // werden nur benötigt wenn versucht werden soll eine Länge zu unterschreiten, z.B. die optimale
            solisopt = false;                   //
            N = anzAnts;
            seed = time(NULL);
            // Es folgen die allocierungen und inizialisierungen
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
            lenlist = (int *)malloc(N*sizeof(int));
            route = (int *)malloc(N*cldim*sizeof(int));
            for (int i = 0; i < cldim; i++) {
                vbestwaysofar.push_back(i);
            }
            lenofbestwaysofar = calulate_way_from_route(vbestwaysofar);
        }
        void initialisiereGPU() {  // cudaMallocs, cudaCopy und setupCernel
            cudaMalloc(&d_state, N*sizeof(curandState));
            block_size = cldim; 
            blocks = N; 
            setup_kernel<<<blocks,block_size>>>(d_state, N, seed);

            cudaMalloc((void **) &d_cost, cldim*cldim*sizeof(int));
            cudaMemcpy(d_cost, cost, cldim*cldim*sizeof(int), cudaMemcpyHostToDevice);
            cudaMalloc((void **) &d_phero, cldim*cldim*sizeof(float));
            cudaMemcpy(d_phero, phero, cldim*cldim*sizeof(float), cudaMemcpyHostToDevice);
            cudaMalloc((void **) &d_lenlist, N*sizeof(int));
            cudaMalloc((void **) &d_route, N*cldim*sizeof(int));
        }
        void oneIteration(float p) {
            //kernel call
            tour_konstruktions_kernel<<<blocks, block_size, cldim*sizeof(int)+cldim*sizeof(float)+cldim*sizeof(bool)>>>(d_state, N, cldim, alpha, beta, d_cost, d_phero, d_route);
            cudaDeviceSynchronize();
            pheromon_evaporation_kernel<<<cldim, cldim>>>(p,d_phero);
            cudaDeviceSynchronize(); //?
            pheromon_aktualisierungs_kernel<<<blocks, block_size, cldim*sizeof(int)>>>(cldim, d_lenlist, d_cost, d_phero, d_route);
            cudaMemcpy(lenlist, d_lenlist, N*sizeof(int), cudaMemcpyDeviceToHost);

            int min = 0;
            for (int i = 1; i < N; i++) if (lenlist[i] < lenlist[min]) min = i; // sucht kleinste länge

            if (lenlist[min] < lenofbestwaysofar){      // wenn die kleinste länge kleiner als die momentan Kleiste Tour ist wird diese überschrieben
                cudaMemcpy(route, d_route, N*cldim*sizeof(int), cudaMemcpyDeviceToHost);
                vbestwaysofar.clear();
                for (int i = 0; i < cldim; i++) {
                    vbestwaysofar.push_back(route[min * cldim + i]);
                }
                lenofbestwaysofar = lenlist[min];
                if (lenofbestwaysofar <= lenofbestway) {
                    solisopt = true; 
                }
            }
        }
        int calulate_way_from_route(vector<int> route) { // utility
            int way = 0;
            for (int i = 0; i < cldim-1; i++) {
                way += cost[route[i]*cldim+route[i+1]];
            }
            way += cost[route[cldim-1]*cldim+route[0]];
            return way;
        }
    public:
        ac(vector<pair<float, float>> citylist, float lenofbestroute=0, int anzAnts=2048) : gen(rd()) { //constructor
            inizialisiereACO(citylist, lenofbestroute, anzAnts);
            initialisiereGPU();
        }
        void doIteration(float p=0.5) {
            oneIteration(p);
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
        void freeall(void) {
            free(cost);
            free(phero);
            free(lenlist);
            cudaFree(d_state);
            cudaFree(d_cost);
            cudaFree(d_phero);
            cudaFree(d_route);
            cudaFree(d_lenlist);
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
            vector<pair<float, float>> citylits = lin318;
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

    return 0;
*/

    // Berechnung der Durchschnittszeit
    for (int i = 0; i < coloniesize.size(); i++) {

        int anzberechungen = 30;
        vector<pair<float, float>> citylits = d198;
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

        // bestrout = region.getbestroute();
        // cout << "bestroute: [";
        // for (const auto& element : bestrout) {
        //     cout << element << ", ";
        // }
        // cout << endl;

        region.freeall();

        float summe = 0.0;
        for (int j = 0; j < anzberechungen; j++) {
            summe += listofdurations[j].count();
        }

        float avg = summe / listofdurations.size();
        cout << "Die Durchschnittliche Ausfuehrungszeit fuer " << coloniesize[i] << " betraegt: " << avg << " Sekunden." << endl;
        // cout << "suration values: ";
        // for (const auto& element : listofdurations) {
        //     cout << element.count() << " ";
        // }
        // cout << endl;
    }

    return 0;
}
