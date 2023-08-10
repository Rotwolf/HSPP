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
        curand_init(seed, ameisenid, 0, &state[ameisenid]);                   // inizialisiert in jedem Block einen eigenen Zufallszahlengenerator
    }
}

/*
__global__ void tour_konstruktions_kernelR(
    curandState *my_curandstate, 
    int N, 
    int cldim, 
    float alpha,
    float beta,
    int *cost, 
    float *phero,
    int *d_route
    ) {

    extern __shared__ int s[];
    int* route = s;
    float* probabilities = (float*)&route[cldim];
    bool* visited = (bool*)&probabilities[cldim];
    __shared__ float sum;

    int ameisenid = blockIdx.x;
    int stadtid = threadIdx.x;

    if (stadtid < cldim) {
        if (stadtid == 0) {
            for (int i = 0; i < cldim; i++) visited[i] = false;
            float myrandstart = curand_uniform(my_curandstate+ameisenid);
            myrandstart *= (cldim -1 +0.99999);
            int start = (int)truncf(myrandstart);
            route[0] = start;
            visited[start] = true;
        }
        __syncthreads();
        for (int i = 1; i < cldim-1; i++) {
            int aktuellestadt = route[i-1];
            
            int anzvs = 0;
            for (int j = 0; j < cldim; j++) {
                if (visited[j]) anzvs++; 
            }
            if (anzvs != i) {
                if (aktuellestadt == -1) {
                    if (0==(int)truncf(sum*1000)) {
                        route[stadtid] = -3;
                    } else {
                        route[stadtid] = (int)(sum*10); 
                    }
                } else {
                    route[stadtid] = -2;
                }
                break;
            }
            
            if (!visited[stadtid]) {
                probabilities[stadtid] = __powf(phero[aktuellestadt*cldim+stadtid]+0.1E-28, alpha) * __powf(1./cost[aktuellestadt*cldim+stadtid], 5);
            } else {
                probabilities[stadtid] = 0;
            }
            __syncthreads();
            if (stadtid == 0) {
                sum = 0;
                for (int j = 0; j < cldim; j++) {
                    sum += probabilities[j];
                }
            }
            __syncthreads();
            probabilities[stadtid] /= sum;
            __syncthreads();
            if (stadtid == 0) {
                float r = curand_uniform(my_curandstate+ameisenid);
                //For (0,1]
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
                    next = j;
                }
                //if (next == -1) sum = sum_prob;
                route[i] = next;
                visited[next] = true;
            }
            __syncthreads();
        }

        if (stadtid == 0) {
            int i = 0;
            while (visited[i]) i++;
            route[cldim-1] = i;
            //visited[i] = true;
            for (int i = 0; i < cldim; i++) d_route[ameisenid*cldim+i] = route[i];
        }
    }
}
*/

__global__ void tour_konstruktions_kernel(
    curandState *my_curandstate, 
    int N, 
    int cldim, 
    float alpha,
    float beta,
    int *cost, 
    float *phero,
    int *d_route
    ) {

    extern __shared__ int s[];
    int* route = s;
    float* probabilities = (float*)&route[cldim];
    bool* visited = (bool*)&probabilities[cldim];
    __shared__ float sum;

    int ameisenid = blockIdx.x;
    int stadtid = threadIdx.x;

    if (stadtid < cldim) {
        if (stadtid == 0) {
            for (int i = 0; i < cldim; i++) visited[i] = false;
            float myrandstart = curand_uniform(my_curandstate+ameisenid);
            myrandstart *= (cldim -1 +0.99999); // its very important that this is not even one 9 longer, since above .999999 the Computer would roud up, leading to an Error
            int start = (int)truncf(myrandstart);
            route[0] = start;
            visited[start] = true;
        }
        __syncthreads();
        for (int i = 1; i < cldim-1; i++) {
            int aktuellestadt = route[i-1];
            if (!visited[stadtid]) {
                probabilities[stadtid] = __powf(phero[aktuellestadt*cldim+stadtid]+0.1E-28, alpha) * __powf(1./(cost[aktuellestadt*cldim+stadtid]+0.1E-3), beta);
            } else {
                probabilities[stadtid] = 0;
            }
            __syncthreads();
            if (stadtid == 0) {
                sum = 0;
                for (int j = 0; j < cldim; j++) {
                    sum += probabilities[j];
                }
            }
            __syncthreads();
            probabilities[stadtid] /= sum;
            __syncthreads();
            if (stadtid == 0) {
                float r = curand_uniform(my_curandstate+ameisenid);
                //For (0,1]
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
                    next = j;
                }
                route[i] = next;
                visited[next] = true;
            }
            __syncthreads();
        }

        if (stadtid == 0) {
            int i = 0;
            while (visited[i]) i++;
            route[cldim-1] = i;
            //visited[i] = true;
            for (int i = 0; i < cldim; i++) d_route[ameisenid*cldim+i] = route[i];
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
    if (schrittid == 0) {
        stadt_i = route[ameisenid*cldim + cldim - 1];
        stadt_j = route[ameisenid*cldim];
    } else {
        stadt_i = route[ameisenid*cldim + schrittid - 1];
        stadt_j = route[ameisenid*cldim + schrittid];
    }

    partial_len[schrittid] = cost[stadt_i * cldim + stadt_j];
    __syncthreads();
    if (schrittid == 0) {
        int len = 0;
        for (int i = 0; i < cldim; i++) len += partial_len[i];
        //write it down, analyse it later.
        lenlist[ameisenid] = len; 
        /*
        if (len < lenofbestwaysofar) {
            for (int i = 0; i < cldim; i++) {
                bestroute[i] = route[ameisenid*cldim + i];  // I'm a Problem
            }
            lenofbestwaysofar = len;
        }
        */
        pherodelta = 1./len;
    }
    __syncthreads();
    atomicAdd(&phero[stadt_i * cldim + stadt_j], pherodelta);
    atomicAdd(&phero[stadt_j * cldim + stadt_i], pherodelta);
}

/*
__global__ void pheromon_aktualisierungs_kernel_old(
    int N, 
    int cldim,
    int lenofbestwaysofar,
    int *bestroute,
    float *cost, 
    float *phero,
    int *route
    ) {

    for (int antid = 0; antid < N; antid++) {
        float len = 0;
        for (int i = 0; i < cldim-1; i++) {
            len += cost[route[antid*cldim + i]*cldim+route[antid*cldim + i+1]];
        }
        len += cost[route[antid*cldim + cldim-1]*cldim+route[antid*cldim]];
        
        if (len < lenofbestwaysofar) {
            for (int i = 0; i < cldim; i++) {
                bestroute[i] = route[antid*cldim + i];
            }
            lenofbestwaysofar = len;
        }

        float nlen = 1/len;

        for (int i = 0; i < cldim-1; i++) {
            phero[route[antid*cldim + i]*cldim+route[antid*cldim + i+1]] += nlen;
            phero[route[antid*cldim + i+1]*cldim+route[antid*cldim + i]] += nlen;
        }
        phero[route[antid*cldim + cldim-1]*cldim+route[antid*cldim]] += nlen;
        phero[route[antid*cldim]*cldim+route[antid*cldim + cldim-1]] += nlen;   
    }
}
*/

vector<pair<float, float>> parseTSPFile(const string& filenameshort) {
    vector<pair<float, float>> coordinates;

    string filename = "C:/Users/jonie/Desktop/hausarbeit/testtsps/" + filenameshort +".tsp";
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
        int *cost;
        int *d_cost;
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
            alpha = 1;
            beta = 2;
            lenofbestwaysofar = 0;
            lenofbestway = lenofbestroute;
            solisopt = false;
            N = anzAnts;
            seed = time(NULL);
            //cout << seed << endl;
            
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
            /*
            for (int x = 0; x < cldim; x++) {
                for (int y = x+1; y < cldim; y++) { // prüfen ob starten komplett mit 0 etwas ändert : ähhh ja, viel besser als 1
                    phero[x*cldim+y] = 0;
                    phero[y*cldim+x] = 0;
                }
            }
            for (int x = 0; x < cldim; x++) phero[x*cldim+x] = 0;
            */
            lenlist = (int *)malloc(N*sizeof(int));
            route = (int *)malloc(N*cldim*sizeof(int));
            for (int i = 0; i < cldim; i++) {
                vbestwaysofar.push_back(i);
            }
            lenofbestwaysofar = calulate_way_from_route(vbestwaysofar);
        }
        void initialisiereGPU() {
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
            for (int i = 1; i < N; i++) if (lenlist[i] < lenlist[min]) min = i;

            if (lenlist[min] < lenofbestwaysofar){
                cudaMemcpy(route, d_route, N*cldim*sizeof(int), cudaMemcpyDeviceToHost);
                vbestwaysofar.clear();
                for (int i = 0; i < cldim; i++) {
                    vbestwaysofar.push_back(route[min * cldim + i]);
                }
                lenofbestwaysofar = lenlist[min];
                if (lenofbestwaysofar <= lenofbestway) {
                    solisopt = true; 
                }
                /*
                cout << "[Nullte len nach para]: " << lenlist[min] << endl;
                cout << "[Nullte len nach seri]: " << calulate_way_from_route(vbestwaysofar) << endl;
                */
            }

            
            /*
            cout << "lenlist: " << endl << "[";
            for (int i = 0; i < N; i++) {
                cout << " " << lenlist[i] << ",";
            }
            cout << "]" << endl;
            */
            /*
            cudaMemcpy(cost, d_cost, cldim*cldim*sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(phero, d_phero, cldim*cldim*sizeof(float), cudaMemcpyDeviceToHost);
            cout << "cost: " << endl << "[";
            for (int i = 0; i < cldim*cldim; i++) {
                cout << " " << cost[i] << ",";
            }
            cout << "]" << endl << "phero: " << endl << "[";
            for (int i = 0; i < cldim*cldim; i++) {
                cout << " " << phero[i] << ",";
            }
            cout << "]" << endl;
            */
            /*
            cudaMemcpy(route, d_route, N*cldim*sizeof(int), cudaMemcpyDeviceToHost);
            for (int i = 0; i < N*cldim; i++) {
                if (route[i] == -1){
                    cout << "[Fehler]:" << endl << "[";
                    for (int j = i; j < i+cldim; j++) {
                        cout << " " << route[j] << ",";
                    }
                    cout << "]" << endl;
                }
            }
            */
            /*
            cout << "route: " << endl << "[";
            for (int i = 0; i < N*cldim; i++) {
                cout << " " << route[i] << ",";
            }
            cout << "]" << endl;
            */
        }
        int calulate_way_from_route(vector<int> route) {
            int way = 0;
            for (int i = 0; i < cldim-1; i++) {
                way += cost[route[i]*cldim+route[i+1]];
            }
            way += cost[route[cldim-1]*cldim+route[0]];
            return way;
        }
    public:
        ac(vector<pair<float, float>> citylist, float lenofbestroute=0, int anzAnts=2048) : gen(rd()) {
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
    //vector<pair<float, float>> cl1 = {{182,663},{232,33},{230,787},{370,676},{256,996},{600,247},{33,672},{119,225},{525,985},{716,397}}; //(3, 8, 4, 2, 0, 6, 7, 1, 5, 9) or [1, 7, 6, 0, 2, 4, 8, 3, 9, 5]
    
    vector<pair<float, float>> dj38 = parseTSPFile("dj38");//{{11003.611100, 42102.500000},{11108.611100, 42373.888900},{11133.333300, 42885.833300},{11155.833300, 42712.500000},{11183.333300, 42933.333300},{11297.500000, 42853.333300},{11310.277800, 42929.444400},{11416.666700, 42983.333300},{11423.888900, 43000.277800},{11438.333300, 42057.222200},{11461.111100, 43252.777800},{11485.555600, 43187.222200},{11503.055600, 42855.277800},{11511.388900, 42106.388900},{11522.222200, 42841.944400},{11569.444400, 43136.666700},{11583.333300, 43150.000000},{11595.000000, 43148.055600},{11600.000000, 43150.000000},{11690.555600, 42686.666700},{11715.833300, 41836.111100}, {11751.111100, 42814.444400},{11770.277800, 42651.944400},{11785.277800, 42884.444400},{11822.777800, 42673.611100},{11846.944400, 42660.555600},{11963.055600, 43290.555600},{11973.055600, 43026.111100},{12058.333300, 42195.555600},{12149.444400, 42477.500000},{12286.944400, 43355.555600},{12300.000000, 42433.333300},{12355.833300, 43156.388900},{12363.333300, 43189.166700},{12372.777800, 42711.388900},{12386.666700, 43334.722200},{12421.666700, 42895.555600},{12645.000000, 42973.333300}};// (12 14 19 22 24 25 21 23 27 26 30 35 33 32 36 34 31 29 28 20 13 9 0 0 1 3 2 4 5 6 7 8 10 11 15 16 17 18)
    int soldj38 = 6656;

    vector<pair<float, float>> lu980 = parseTSPFile("lu980");;
    int sollu980 = 11340;

    vector<pair<float, float>> qa194 = parseTSPFile("qa194");;
    int solqa194 = 9352;

    vector<pair<float, float>> a280 = parseTSPFile("a280");//{{288, 149}, {288, 129}, {270, 133}, {256, 141}, {256, 157}, {246, 157}, {236, 169}, {228, 169}, {228, 161}, {220, 169}, {212, 169}, {204, 169}, {196, 169}, {188, 169}, {196, 161}, {188, 145}, {172, 145}, {164, 145}, {156, 145}, {148, 145}, {140, 145}, {148, 169}, {164, 169}, {172, 169}, {156, 169}, {140, 169}, {132, 169}, {124, 169}, {116, 161}, {104, 153}, {104, 161}, {104, 169}, {90, 165}, {80, 157}, {64, 157}, {64, 165}, {56, 169}, {56, 161}, {56, 153}, {56, 145}, {56, 137}, {56, 129}, {56, 121}, {40, 121}, {40, 129}, {40, 137}, {40, 145}, {40, 153}, {40, 161}, {40, 169}, {32, 169}, {32, 161}, {32, 153}, {32, 145}, {32, 137}, {32, 129}, {32, 121}, {32, 113}, {40, 113}, {56, 113}, {56, 105}, {48, 99}, {40, 99}, {32, 97}, {32, 89}, {24, 89}, {16, 97}, {16, 109}, {8, 109}, {8, 97}, {8, 89}, {8, 81}, {8, 73}, {8, 65}, {8, 57}, {16, 57}, {8, 49}, {8, 41}, {24, 45}, {32, 41}, {32, 49}, {32, 57}, {32, 65}, {32, 73}, {32, 81}, {40, 83}, {40, 73}, {40, 63}, {40, 51}, {44, 43}, {44, 35}, {44, 27}, {32, 25}, {24, 25}, {16, 25}, {16, 17}, {24, 17}, {32, 17}, {44, 11}, {56, 9}, {56, 17}, {56, 25}, {56, 33}, {56, 41}, {64, 41}, {72, 41}, {72, 49}, {56, 49}, {48, 51}, {56, 57}, {56, 65}, {48, 63}, {48, 73}, {56, 73}, {56, 81}, {48, 83}, {56, 89}, {56, 97}, {104, 97}, {104, 105}, {104, 113}, {104, 121}, {104, 129}, {104, 137}, {104, 145}, {116, 145}, {124, 145}, {132, 145}, {132, 137}, {140, 137}, {148, 137}, {156, 137}, {164, 137}, {172, 125}, {172, 117}, {172, 109}, {172, 101}, {172, 93}, {172, 85}, {180, 85}, {180, 77}, {180, 69}, {180, 61}, {180, 53}, {172, 53}, {172, 61}, {172, 69}, {172, 77}, {164, 81}, {148, 85}, {124, 85}, {124, 93}, {124, 109}, {124, 125}, {124, 117}, {124, 101}, {104, 89}, {104, 81}, {104, 73}, {104, 65}, {104, 49}, {104, 41}, {104, 33}, {104, 25}, {104, 17}, {92, 9}, {80, 9}, {72, 9}, {64, 21}, {72, 25}, {80, 25}, {80, 41}, {88, 49}, {104, 57}, {124, 69}, {124, 77}, {132, 81}, {140, 65}, {132, 61}, {124, 61}, {124, 53}, {124, 45}, {124, 37}, {124, 29}, {132, 21}, {124, 21}, {120, 9}, {128, 9}, {136, 9}, {148, 9}, {162, 9}, {156, 25}, {172, 21}, {180, 21}, {180, 29}, {172, 29}, {172, 37}, {172, 45}, {180, 45}, {180, 37}, {188, 41}, {196, 49}, {204, 57}, {212, 65}, {220, 73}, {228, 69}, {228, 77}, {236, 77}, {236, 69}, {236, 61}, {228, 61}, {228, 53}, {236, 53}, {236, 45}, {228, 45}, {228, 37}, {236, 37}, {236, 29}, {228, 29}, {228, 21}, {236, 21}, {252, 21}, {260, 29}, {260, 37}, {260, 45}, {260, 53}, {260, 61}, {260, 69}, {260, 77}, {276, 77}, {276, 69}, {276, 61}, {276, 53}, {284, 53}, {284, 61}, {284, 69}, {284, 77}, {284, 85}, {284, 93}, {284, 101}, {288, 109}, {280, 109}, {276, 101}, {276, 93}, {276, 85}, {268, 97}, {260, 109}, {252, 101}, {260, 93}, {260, 85}, {236, 85}, {228, 85}, {228, 93}, {236, 93}, {236, 101}, {228, 101}, {228, 109}, {228, 117}, {228, 125}, {220, 125}, {212, 117}, {204, 109}, {196, 101}, {188, 93}, {180, 93}, {180, 101}, {180, 109}, {180, 117}, {180, 125}, {196, 145}, {204, 145}, {212, 145}, {220, 145}, {228, 145}, {236, 145}, {246, 141}, {252, 125}, {260, 129}, {280, 133}};
    int sola280 = 2579;
    
    vector<pair<float, float>> d198 = parseTSPFile("d198");;
    int sold198 = 15780;

    vector<pair<float, float>> lin318 = parseTSPFile("lin318");;
    int sollin318 = 42029;

    vector<pair<float, float>> pcb442 = parseTSPFile("pcb442");;
    int solpcb442 = 50778;

    vector<pair<float, float>> pr1002 = parseTSPFile("pr1002");;
    int solpr1002 = 259045;

    vector<pair<float, float>> rat783 = parseTSPFile("rat783");;
    int solrat783 = 8806;


    vector<int> coloniesize = {1024, 2048, 4096, 8192}; //8192,4096,2048,1024

    for (int i = 0; i < coloniesize.size(); i++) {

        int anzberechungen = 30;
        vector<pair<float, float>> citylits = lin318;
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
        /*
        bestrout = region.getbestroute();
        cout << "bestroute: [";
        for (const auto& element : bestrout) {
            cout << element << ", ";
        }
        cout << endl;
        */
        region.freeall();

        float summe = 0.0;
        for (int j = 0; j < anzberechungen; j++) {
            summe += listofdurations[j].count();
        }

        float avg = summe / listofdurations.size();
        cout << "Die Durchschnittliche Ausfuehrungszeit fuer " << coloniesize[i] << " betraegt: " << avg << " Sekunden." << endl;
        /*
        cout << "suration values: ";
        for (const auto& element : listofdurations) {
            cout << element.count() << " ";
        }
        cout << endl;
        */
    }

    return 0;

/*
    vector<chrono::duration<float>> listofdurations;
    int anzberechungen = 30;
    int maxlastbestroutechange = 100000;
    vector<pair<float, float>> citylits = dj38;
    int lenofbesttour = soldj38;
    listofdurations.resize(anzberechungen); 

    for (int j = 0; j < anzberechungen; j++) {
        vector<int> bestrout;
        int bestroutlen = INT_MAX;
        int newbestroutlen;
        int lastbestroutechange = 0;
        ac region(citylits, lenofbesttour, 1024); //8192,4096,2048,1024,256  // Change the used TSP-Instance here (and dont forget to change the soltion length: solxxxx)
        //ac region(qa194, solqa194, 1024);

        auto start = chrono::high_resolution_clock::now();

        int i = -1;
        while (!region.issolopt() && lastbestroutechange<maxlastbestroutechange) {
            i++;
            //cout <<  i << endl; 
            region.doIteration(0.5);

            newbestroutlen = region.getbestroutelen();
            cout << "bestroutlen: " << newbestroutlen << endl;
            if (newbestroutlen < bestroutlen) {
                bestroutlen = newbestroutlen;
                lastbestroutechange = 0;
            } else {
                lastbestroutechange++;
            }
            /*
            cout << "lastchange was: * " << lastbestroutechange << " * Iterations ago." << endl;
            bestrout = region.getbestroute();
            if (true) { // true // newbestroutlen < lenofbesttour
                cout << "bestroute: [";
                for (const auto& element : bestrout) {
                    cout << element << ", ";
                }
                cout << endl;
            }
            */
/*
            if (newbestroutlen > bestroutlen) {
                break;
            }
            
            if (lastbestroutechange >= maxlastbestroutechange) {
                cout << "[SAD] ACO broke because of the max iterations when bestrout doesnt change." << endl;
            }
        }
        
        bestroutlen = region.getbestroutelen();
        cout << "bestroutelen: " << bestroutlen << endl;
        bestrout = region.getbestroute();
        cout << "bestroute: [";
        for (const auto& element : bestrout) {
            cout << element << ", ";
        }
        cout << endl;
        /*
        cout << "sorted bestroute: ";
        sort(bestrout.begin(), bestrout.end());
        for (const auto& element : bestrout) {
            cout << element << " ";
        }
        cout << endl;
        */
/*
        region.freeall();
        
        auto end = chrono::high_resolution_clock::now();
        listofdurations[j] = end - start;

        cout << "[" << j+1 << "/" << anzberechungen << "]" << endl;
    }
    
    float summe = 0.0;
    for (int i = 0; i < anzberechungen; i++) {
        summe += listofdurations[i].count();
    }
    
    float avg = summe / listofdurations.size();
    cout << "Die Durchschnittliche Ausfuehrungszeit betraegt: " << avg << " Sekunden." << endl;

    return 0;
*/
}   

