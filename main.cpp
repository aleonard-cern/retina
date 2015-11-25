// different precompiler flags:
#define DEBUG	        2 // 0 does not print anything
                          // 1 prints the generated tracks, the reconstructed track
                          // 2 same as 1 plus calls gnuplot to output the retina response matrix
#define USEEXPARRAY		1 // 0 will compute the exponential for every distances
                          // 1 will use a precomputed array of exponential values
#define USERANDOMSEED	0 // 0 uses a fixed seed for the random generator
                          // 1 uses a time-based seed
#define MULTITHREAD     2 // 0 no multithreading
                          // 1 four thread created for every event
                          // 2 four threaded created in a thread-pool
#define SSE				1 // 0 does not vectorize the code
                          // 1 uses SIM technology to vectorize the code

#define PI 3.14159265359 // this is at least portable ....
#if MULTITHREAD == 1
#include <thread>
#endif
#if MULTITHREAD == 2
#include "ctpl_stl.h"
#include <future>
#endif
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#ifdef _WIN32
#include <direct.h>
#else
#include <unistd.h>
#endif



class Layer
{
    public:
        Layer(){
            this->z = 0;
            this->xNBins = 0;
            this->yNBins = 0;
            xBins = nullptr;
            yBins = nullptr;
        }
        ~Layer(){
            delete[] xBins;
            xBins = nullptr;
            delete[] yBins;
            yBins = nullptr;
        }

        void setLayer(int xNBins, int yNBins, int zCoord) {
            this->z = zCoord;
            this->xNBins = xNBins;
            this->yNBins = yNBins;

            xBins = new int[xNBins + 1];
            for (int i = 0; i < xNBins; i++) {
                xBins[i] = -2000 + i * 4000 / xNBins;
            }

            yBins = new int[yNBins + 1];
            for (int i = 0; i < yNBins; i++) {
                yBins[i] = -2000 + i * 4000 / yNBins;
            }
        }
        int getZCoord(){ return z; }

        int xNBins;
        int yNBins;
        int* xBins;
        int* yBins;

        int z;

};

// create global exponential array for argument from 0 to -5 by steps of 1./128.
unsigned int expArray[640];
void initExpArray() {
    for (int i = 0; i < 640; ++i) {
        expArray[i] = 512 * exp(-i / 128.0);
    }
}

// structure of recoHit consisting of 3 integer coordinates
struct RecoHit
{
    int x, y, z;
};


// This function contains the actual RETINA algorithm
// the first argument (id) is used only for the thread-pool
inline void computeRetinaResponse(int id, int start, int end, int nLayers, int* response, int rNReco, RecoHit* recoHits, int** mappingX, int** mappingY) {

#if SSE
    int index[4] = { 0, 0, 0, 0 };
#else
    int index = 0;
    int dist2 = 0;
#endif // end if SSE
    for (int i = 0; i < rNReco; ++i) { // start rNReco loop

#if SSE

#ifdef __unix__	
        asm volatile (
                "movd %0, %%xmm1              \n\t" // load recoHit x to xmm1
                "pshufd $0x00, %%xmm1, %%xmm1 \n\t"
                "movd %1, %%xmm2              \n\t" // load recoHit y to xmm2
                "pshufd $0x00, %%xmm2, %%xmm2 \n\t" 
                : : "m" (recoHits[i].x), "m" (recoHits[i].y) : "cc"
                );
#endif // end if __unix__

#ifdef _WIN32
        int recoX = recoHits[i].x;
        int recoY = recoHits[i].y;
#endif // end if _WIN32

        // increment by four because we dot 4 cells at a time thanks to SSE vectorization
        for (int j = start; j < end; j += 4) {
#ifdef __unix__	
            asm volatile (
                    "movdqu %1, %%xmm3 \n\t" // load nominal tracks x
                    "movdqu %2, %%xmm4 \n\t" // load nominal tracks y

                    "psubd %%xmm1, %%xmm3 \n\t" // subtract reco x from nominal x
                    "psubd %%xmm2, %%xmm4 \n\t" // subtract reco x from nominal x

                    "pmulld %%xmm3, %%xmm3 \n\t" // square each difference
                    "pmulld %%xmm4, %%xmm4 \n\t" // square each difference

                    "paddd %%xmm4, %%xmm3 \n\t" // add the x and y component
                    "psrld $0x04, %%xmm3 \n\t" // shift by 4 --> divide by 16 = 128 / 2048 to fit the expArray
                    "movdqu %%xmm3, %0 \n\t" // store it in index array 
                    : "=m" (index) : "m" (mappingX[recoHits[i].z][j]), "m" (mappingY[recoHits[i].z][j]) : "cc"
                    );
#endif // end if __unix__

#ifdef _WIN32
            int nomX[4], nomY[4];
            for (int k = 0; k < 4; ++k) {
                nomX[k] = mappingX[recoHits[i].z][j + k];
                nomY[k] = mappingY[recoHits[i].z][j + k];
            }

            __asm {

                MOVD XMM1, recoX // load recoHit x to xmm1
                    PSHUFD XMM1, XMM1, 0x00
                    MOVD XMM2, recoY // load recoHit y to xmm2
                    PSHUFD XMM2, XMM2, 0x00

                    MOVDQA XMM3, nomX // load nominal tracks x
                    MOVDQA XMM4, nomY // load nominal tracks y

                    PSUBD XMM3, XMM1 // subtract reco x from nominal x
                    PSUBD XMM4, XMM2 // subtract reco x from nominal x

                    PMULLD XMM3, XMM3 // square each difference
                    PMULLD XMM4, XMM4 // square each difference

                    PADDD XMM3, XMM4 // add the x and y component
                    PSRLD XMM3, 0x04 // shift by 4 --> divide by 16 = 128 / 2048 to fit the expArray
                    MOVDQA index, XMM3 // store it in index array 
            }
#endif // end if _WIN32

#if USEEXPARRAY
            for (int k = 0; k < 4; ++k) {

                // using the precomputed exparray
                //index[k] = (int)(128 * dist2[k] / 2048);
                if (index[k] >= 0 && index[k] < 640) {
                    response[j + k] += expArray[index[k]];
                }
            }
#else // do not use precomputed exp array
            for (int k = 0; k < 4; ++k) {
                // computing the exponential in the loop
                response[j + k] += 8192 * exp(-dist2[k] / (2048));
            }
#endif // end if USEEXPARRAY
        } // loop over cell with SSE

#else // No SSE increment by one
        for (int j = start; j < end; j += 1) {

            dist2 = pow(recoHits[i].x - mappingX[recoHits[i].z][j], 2);
            dist2 += pow(recoHits[i].y - mappingY[recoHits[i].z][j], 2);

#if USEEXPARRAY
            // using the precomputed exparray
            index = -(int)(-128 * dist2 / 2048);
            if (index >= 0 && index < 640) {
                response[j] += expArray[index];
            }
#else // do not use precomputed exp array
            // computing the exponential in the loop
            response[j] += 8192 * exp(-dist2 / 2048);
#endif // end if USEEXPARRAY

        } // loop over cell no SSE
#endif // end if SSE
    }
}

// this function goes through the entire RETINA respons matrix to locate the
// multiple local maxima. A local maxima is a cell whose neighouring cells
// are all less or egal. A threshold can be given as third argument.
inline int locateMaxima(int nTheta, int nPhi, int* response, int threshold = 0) {

    int counter = 0;
    // if (i == 0 && j == 0) 
    if (response[0 * nTheta + 0] > threshold &&
            response[0 * nTheta + 0] >= response[0 * nTheta + 1] &&
            response[0 * nTheta + 0] >= response[1 * nTheta + 0] &&
            response[0 * nTheta + 0] >= response[1 * nTheta + 1]) {
#if DEBUG
        std::cout << std::setw(4) << counter << ": theta = " << std::setw(6) << -30 + 0 * 60.0 / (nTheta - 1) << "\tphi = " << -30 + 0 * 60.0 / (nPhi - 1) << std::endl;
#endif
        counter++;
    }

    // if (i == 0 && j == ySize - 1) 
    if (response[(nPhi - 1) * nTheta + 0] > threshold &&
            response[(nPhi - 1) * nTheta + 0] >= response[(nPhi - 1) * nTheta + 1] &&
            response[(nPhi - 1) * nTheta + 0] >= response[(nPhi - 2) * nTheta + 0] &&
            response[(nPhi - 1) * nTheta + 0] >= response[(nPhi - 2) * nTheta + 1]) {
#if DEBUG
        std::cout << std::setw(4) << counter << ": theta = " << std::setw(6) << -30 + 0 * 60.0 / (nTheta - 1) << "\tphi = " << -30 + (nPhi - 1) * 60.0 / (nPhi - 1) << std::endl;
#endif
        counter++;
    }

    // if (i == xSize - 1 && j == 0) 
    if (response[0 * nTheta + (nTheta - 1)] > threshold &&
            response[0 * nTheta + (nTheta - 1)] >= response[1 * nTheta + (nTheta - 1)] &&
            response[0 * nTheta + (nTheta - 1)] >= response[0 * nTheta + (nTheta - 2)] &&
            response[0 * nTheta + (nTheta - 1)] >= response[1 * nTheta + (nTheta - 2)]) {
#if DEBUG
        std::cout << std::setw(4) << counter << ": theta = " << std::setw(6) << -30 + (nTheta - 1) * 60.0 / (nTheta - 1) << "\tphi = " << -30 + 0 * 60.0 / (nPhi - 1) << std::endl;
#endif
        counter++;
    }

    // if (i == xSize - 1 && j == ySize - 1) 
    if (response[(nPhi - 1) * nTheta + (nTheta - 1)] > threshold &&
            response[(nPhi - 1) * nTheta + (nTheta - 1)] >= response[(nPhi - 2) * nTheta + (nTheta - 1)] &&
            response[(nPhi - 1) * nTheta + (nTheta - 1)] >= response[(nPhi - 1) * nTheta + (nTheta - 2)] &&
            response[(nPhi - 1) * nTheta + (nTheta - 1)] >= response[(nPhi - 2) * nTheta + (nTheta - 2)]) {
#if DEBUG
        std::cout << std::setw(4) << counter << ": theta = " << std::setw(6) << -30 + (nTheta - 1) * 60.0 / (nTheta - 1) << "\tphi = " << -30 + (nPhi - 1) * 60.0 / (nPhi - 1) << std::endl;
#endif
        counter++;
    }

    // if (i == 0 && j != 0 && j=! ySize - 1)
    for (int j = 1; j < nPhi - 2; ++j) {
        if (response[j * nTheta + 0] > threshold &&
                response[j * nTheta + 0] >= response[j * nTheta + 1] &&
                response[j * nTheta + 0] >= response[(j + 1) * nTheta + 1] &&
                response[j * nTheta + 0] >= response[(j - 1) * nTheta + 1] &&
                response[j * nTheta + 0] >= response[(j + 1) * nTheta + 0] &&
                response[j * nTheta + 0] >= response[(j - 1) * nTheta + 0]) {
#if DEBUG
            std::cout << std::setw(4) << counter << ": theta = " << std::setw(6) << -30 + 0 * 60.0 / (nTheta - 1) << "\tphi = " << -30 + j * 60.0 / (nPhi - 1) << std::endl;
#endif
            counter++;
        }
    }
    // if (i == xSize -1 && j != 0 && j! ySize - 1)
    for (int j = 1; j < nPhi - 2; ++j) {
        if (response[j * nTheta + (nTheta - 1)] > threshold &&
                response[j * nTheta + (nTheta - 1)] >= response[j * nTheta + (nTheta - 2)] &&
                response[j * nTheta + (nTheta - 1)] >= response[(j + 1) * nTheta + (nTheta - 2)] &&
                response[j * nTheta + (nTheta - 1)] >= response[(j - 1) * nTheta + (nTheta - 2)] &&
                response[j * nTheta + (nTheta - 1)] >= response[(j + 1) * nTheta + (nTheta - 1)] &&
                response[j * nTheta + (nTheta - 1)] >= response[(j - 1) * nTheta + (nTheta - 1)]) {
#if DEBUG
            std::cout << std::setw(4) << counter << ": theta = " << std::setw(6) << -30 + (nTheta - 1) * 60.0 / (nTheta - 1) << "\tphi = " << -30 + j * 60.0 / (nPhi - 1) << std::endl;
#endif
            counter++;
        }
    }

    // if (i != 0 && i != xSize - 1 && j == 0)
    for (int i = 1; i < nTheta - 2; ++i) {
        if (response[0 * nTheta + i] > threshold &&
                response[0 * nTheta + i] >= response[1 * nTheta + i] &&
                response[0 * nTheta + i] >= response[1 * nTheta + (i + 1)] &&
                response[0 * nTheta + i] >= response[1 * nTheta + (i - 1)] &&
                response[0 * nTheta + i] >= response[0 * nTheta + (i + 1)] &&
                response[0 * nTheta + i] >= response[0 * nTheta + (i - 1)]) {
#if DEBUG
            std::cout << std::setw(4) << counter << ": theta = " << std::setw(6) << -30 + i * 60.0 / (nTheta - 1) << "\tphi = " << -30 + 0 * 60.0 / (nPhi - 1) << std::endl;
#endif
            counter++;
        }
    }

    // if (i != 0 && i != xSize - 1 && j == ySize - 1)
    for (int i = 1; i < nTheta - 2; ++i) {
        if (response[(nPhi - 1) * nTheta + i] > threshold &&
                response[(nPhi - 1) * nTheta + i] >= response[(nPhi - 2) * nTheta + i] &&
                response[(nPhi - 1) * nTheta + i] >= response[(nPhi - 2) * nTheta + (i + 1)] &&
                response[(nPhi - 1) * nTheta + i] >= response[(nPhi - 2) * nTheta + (i - 1)] &&
                response[(nPhi - 1) * nTheta + i] >= response[(nPhi - 1) * nTheta + (i + 1)] &&
                response[(nPhi - 1) * nTheta + i] >= response[(nPhi - 1) * nTheta + (i - 1)]) {
#if DEBUG
            std::cout << std::setw(4) << counter << ": theta = " << std::setw(6) << -30 + i * 60.0 / (nTheta - 1) << "\tphi = " << -30 + (nPhi - 1) * 60.0 / (nPhi - 1) << std::endl;
#endif
            counter++;
        }
    }

    // if (i != 0 && i != xSize - 1 && j != 0 && j != ySize - 1)
    for (int i = 1; i < nTheta - 1; ++i) {
        for (int j = 1; j < nPhi - 1; ++j) {
            if (response[j * nTheta + i] > threshold &&
                    response[j * nTheta + i] >= response[j * nTheta + (i - 1)] &&
                    response[j * nTheta + i] >= response[j * nTheta + (i + 1)] &&
                    response[j * nTheta + i] >= response[(j + 1) * nTheta + i] &&
                    response[j * nTheta + i] >= response[(j - 1) * nTheta + i] &&
                    response[j * nTheta + i] >= response[(j + 1) * nTheta + (i + 1)] &&
                    response[j * nTheta + i] >= response[(j + 1) * nTheta + (i - 1)] &&
                    response[j * nTheta + i] >= response[(j - 1) * nTheta + (i + 1)] &&
                    response[j * nTheta + i] >= response[(j - 1) * nTheta + (i - 1)]) {
#if DEBUG
                std::cout << std::setw(4) << counter << ": theta = " << std::setw(6) << -30 + i * 60.0 / (nTheta - 1) << "\tphi = " << -30 + j * 60.0 / (nPhi - 1) << std::endl;
#endif
                counter++;
            }
        }
    }

    return counter;
}


int main(int argc, char* argv[])
{
    // number of detector layer
    int nLayers = 6;
#if USEEXPARRAY
    initExpArray();
#endif

    // here we build the detector, consisting in nLayers parallel square planes of 2000 mm side length
    // and containing rectangular sensors of size 1 mm x 100 mm (X - Y) and located at Z = 1200 + i * 100 mm 
    Layer* detectorLayers = new Layer[nLayers];
    for (int i = 0; i < nLayers; ++i) {
        detectorLayers[i].setLayer(2000, 20, 1200 + i * 100); // units are in micro meters
    }

    // phi segmentation covers -30 to 30 degres
    // phi = [-30, -29, ..., -1, 0, 1, ..., 29, 30] 
    int nPhi = 61;
    float* phi = new float[nPhi];
    for (int i = 0; i < nPhi; ++i) {
        phi[i] = -30 + i * 60.0 / (nPhi - 1);
    }

    // theta segmenation same as phi
    // theta = [-30, -29, ..., -1, 0, 1, ..., 29, 30] 
    int nTheta = 61;
    float* theta = new float[nTheta];
    for (int i = 0; i < nTheta; ++i) {
        theta[i] = -30 + i * 60.0 / (nTheta - 1);
    }

    // will add x_0 and y_0 later
    int x_0 = 0;
    int y_0 = 0;


    // create the 2D response
    int realAllocatedSize = ((nPhi * nTheta + 3) / 4) * 4;
    int* response = new int[realAllocatedSize];

    // create the mapping phi, theta, layer
    int** mappingX = new int*[nLayers];
    int** mappingY = new int*[nLayers];
    for (int i = 0; i < nLayers; ++i) {
        mappingX[i] = new int[realAllocatedSize];
        mappingY[i] = new int[realAllocatedSize];
    }

    // compute the impact of the perfect lines
    for (int i_phi = 0; i_phi < nPhi; ++i_phi) {
        for (int i_theta = 0; i_theta < nTheta; ++i_theta) {
            for (int i_layer = 0; i_layer < nLayers; ++i_layer) {
                mappingX[i_layer][i_phi * nTheta + i_theta] = (tan(phi[i_phi] * PI / 180) * detectorLayers[i_layer].z + x_0);
                mappingY[i_layer][i_phi * nTheta + i_theta] = (tan(theta[i_theta] * PI / 180) / cos(phi[i_phi] * PI / 180) * detectorLayers[i_layer].z + y_0);
            }
        }
    }

#if USERANDOMSEED
    srand(time(NULL));
#else
    srand(0);
#endif
    int counter = 0;

    float *rTheta = nullptr;
    float *rPhi = nullptr;
    int *nRecoHits = nullptr;
    RecoHit* recoHits = nullptr;

#if MULTITHREAD == 1 
    std::thread first;
    std::thread second;
    std::thread third;
    std::thread fourth;
#elif MULTITHREAD == 2
    ctpl::thread_pool p(8);
    std::vector<std::future<void>> results(8);
#endif

    // loop on 5000 random events containing a random number of random tracks
    while (counter++ < 50000) {
        // reset the response matrix to 0
        std::fill(response, response + nPhi * nTheta, 0);
        // input reco hits
        // generate random number of tracks
        int rNReco = rand() % 20 + 1;
#if DEBUG
        std::cout << rNReco << " tracks generated with the following angles:" << std::endl;
#endif
        // generate random parameters for the tracks (tracks are straigh line
        // passing by the origin (0, 0, 0) and parameterized by two angles: phi and theta
        rTheta = new float[rNReco];
        rPhi = new float[rNReco];
        for (int i = 0; i < rNReco; ++i) {
            rTheta[i] = (rand() % 600 - 300) * 1.0 / 10;
            rPhi[i] = (rand() % 600 - 300) * 1.0 / 10;
#if DEBUG
            std::cout << std::setw(6) << i << ": theta = " << rTheta[i] << "\tphi = " << rPhi[i] << std::endl;
#endif
        }
#if DEBUG
        std::cout << std::endl;
#endif
        // for now 100% efficiency
        // allocate array of hits
        recoHits = new RecoHit[nLayers * rNReco];
        // construct the reco hits of the generated tracks
        for (int i = 0; i < nLayers; ++i) {
            for (int j = 0; j < rNReco; ++j) {
                recoHits[i * rNReco + j].x = (((int)(tan(rPhi[j] * PI / 180) * detectorLayers[i].z ) ) );
                recoHits[i * rNReco + j].y = (((int)(tan(rTheta[j] * PI / 180) / cos(rPhi[j] * PI / 180) * detectorLayers[i].z + 25) / 50 ) * 50 );
                recoHits[i * rNReco + j].z = i;
            }
        }

        // Now we can compute the Retina response
#if MULTITHREAD == 1
        first = std::thread(computeRetinaResponse, 0, 0, (nPhi*nTheta)/4, nLayers, response, nLayers * rNReco, recoHits, mappingX, mappingY);
        second = std::thread(computeRetinaResponse, 0, (nPhi*nTheta)/4, (nPhi*nTheta)/2, nLayers, response, nLayers * rNReco, recoHits, mappingX, mappingY);
        third = std::thread(computeRetinaResponse, 0, (nPhi*nTheta)/2, 3 * (nPhi*nTheta)/4, nLayers, response, nLayers * rNReco, recoHits, mappingX, mappingY);
        fourth = std::thread(computeRetinaResponse, 0, 3 * (nPhi*nTheta) / 4, nPhi*nTheta, nLayers, response, nLayers * rNReco, recoHits, mappingX, mappingY);

        first.join();
        second.join();
        third.join();
        fourth.join();
#elif MULTITHREAD == 2
        for (int i=0; i < 8; i++) {
            results[i] = p.push(computeRetinaResponse, i*(nPhi*nTheta) / 8, (i+1)*(nPhi*nTheta) / 8, nLayers, response, nLayers * rNReco, recoHits, mappingX, mappingY);
        }
        for (int i=0; i < 8; i++) {
            results[i].get();
        }
#else  
        computeRetinaResponse(0, 0, nPhi*nTheta, nLayers, response, nLayers * rNReco, recoHits, mappingX, mappingY);
#endif

#if DEBUG
        std::cout << "\n The local maxima are situated at entry:" << std::endl;
#endif
        // and try to locate all the local maxima
        int nRecoTracks = locateMaxima(nTheta, nPhi, response, 20);

#if DEBUG
        std::cout << nRecoTracks << " tracks were found." << std::endl;
#endif

#if DEBUG == 2 
        // write to txt file for analysis
        std::string filename = "testInt.txt";
        std::ofstream ofs(filename, std::ofstream::out);

        for (int i_phi = 0; i_phi < nPhi; ++i_phi) {
            for (int i_theta = 0; i_theta < nTheta; ++i_theta) {
                ofs << response[i_phi * nTheta + i_theta] << "\t";
            }
            ofs << std::endl;
        }
        ofs.close();

        // plotting with gnuplot
#ifdef _WIN32
        FILE *pipe = _popen("C:\\Program\" \"Files\\gnuplot\\bin\\gnuplot", "w");
        std::string term = "wx";
#endif
#ifdef __unix__
        FILE *pipe = popen("/usr/bin/gnuplot", "w");
        std::string term = "x11";
#endif
        if (pipe != NULL)
        {
            fprintf(pipe, "set term %s size 850,400\n", term.c_str());         // set the terminal
            fprintf(pipe, "set xrange [-30.5:30.5]\n"); // plot type
            fprintf(pipe, "set yrange [-30.5:30.5]\n"); // plot type
            fprintf(pipe, "set pm3d map\n");
            fprintf(pipe, "set title \"Retina Response\"\n");
            fprintf(pipe, "set xlabel \"Theta (deg)\"\n");
            fprintf(pipe, "set ylabel \"Phi (deg)\"\n");
            fprintf(pipe, "splot \"%s\" u (-30+$1*60/%d):(-30+$2*60/%d):3 matrix w image notitle\n", filename.c_str(), nTheta - 1, nPhi - 1);
            fflush(pipe);                           // flush the pipe

            //system("pause");      // wait for key press
            std::cout << "Do you want more ? Y/N" << std::endl;
            std::cin.clear();
            std::cin.ignore(std::cin.rdbuf()->in_avail());
            char answer = std::cin.get();
            if (answer == 'N' || answer == 'n') {
                fprintf(pipe, "exit\n");
                break;
            }
            fflush(pipe);
#ifdef _WIN32
            _pclose(pipe);
#else
            pclose(pipe);
#endif
        }
        else {
            std::cout << "Could not open pipe" << std::endl;
        }
#endif

#if DEBUG == 1
        std::cout << "Do you want more ? Y/N" << std::endl;
        std::cin.clear();
        std::cin.ignore(std::cin.rdbuf()->in_avail());
        char answer = std::cin.get();
        if (answer == 'N' || answer == 'n') {
            break;
        }
#endif


        delete[] rTheta;
        delete[] rPhi;
        delete[] recoHits;
        delete[] nRecoHits;

    }

    // not sure this is needed ... should do it for every array ...
    // dealocate memories

    for (int i = 0; i < nLayers; ++i) {
        delete[] mappingX[i];
        delete[] mappingY[i];
    }
    delete[] mappingX;
    delete[] mappingY;

    delete[] detectorLayers;
    delete[] phi;
    delete[] theta;

    delete[] response;


    return 0;
}
