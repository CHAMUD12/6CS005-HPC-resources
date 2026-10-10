// 6CS005 - Task 3: File-based password cracking with CUDA
// Input format: one encrypted 10-character password per line.
// Original password format: two lowercase letters followed by two digits.
// Build: nvcc -std=c++14 -O2 password_cracker.cu -o password_cracker
// Run:   ./password_cracker passwords.txt [decrypted.txt]

#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>
#include <vector>
#include <climits>

static const int RAW_LENGTH = 4;
static const int ENCRYPTED_LENGTH = 10;
static const int RAW_STRIDE = RAW_LENGTH + 1;
static const int ENCRYPTED_STRIDE = ENCRYPTED_LENGTH + 1;
static const int TOTAL_CANDIDATES = 26 * 26 * 10 * 10;  // 67,600
static const int THREADS_PER_BLOCK = 256;
static const int GROUPS_PER_PASSWORD =
    (TOTAL_CANDIDATES + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

// Keep all transformations (including the unusual lower-bound correction)
// exactly as they appear in the lecturer-provided CryptForCuda.c.
// Each CUDA thread owns its own output buffer: no shared/static output array.
__device__ void cudaCryptDevice(const char raw[RAW_STRIDE],
                                char encrypted[ENCRYPTED_STRIDE]) {
    int v[ENCRYPTED_LENGTH];
    v[0] = raw[0] + 2;
    v[1] = raw[0] - 2;
    v[2] = raw[0] + 1;
    v[3] = raw[1] + 3;
    v[4] = raw[1] - 3;
    v[5] = raw[1] - 1;
    v[6] = raw[2] + 2;
    v[7] = raw[2] - 2;
    v[8] = raw[3] + 4;
    v[9] = raw[3] - 4;

    for (int i = 0; i < ENCRYPTED_LENGTH; ++i) {
        if (i < 6) {
            if (v[i] > 122) {
                v[i] = (v[i] - 122) + 97;
            } else if (v[i] < 97) {
                v[i] = (97 - v[i]) + 97;
            }
        } else {
            if (v[i] > 57) {
                v[i] = (v[i] - 57) + 48;
            } else if (v[i] < 48) {
                v[i] = (48 - v[i]) + 48;
            }
        }
        encrypted[i] = static_cast<char>(v[i]);
    }
    encrypted[ENCRYPTED_LENGTH] = '\0';
}

// Logical block = one group of up to 256 guesses for one input password.
// Blocks and threads use the x dimension. For very large files, grid-stride
// scheduling covers all logical blocks without exceeding the grid limit.
__global__ void crackPasswords(const char *encryptedInput,
                               char *recoveredOutput,
                               int *foundFlags,
                               int passwordCount) {
    const unsigned long long logicalBlockCount =
        static_cast<unsigned long long>(passwordCount) * GROUPS_PER_PASSWORD;

    for (unsigned long long logicalBlock = blockIdx.x;
         logicalBlock < logicalBlockCount;
         logicalBlock += gridDim.x) {
        const int passwordIndex =
            static_cast<int>(logicalBlock / GROUPS_PER_PASSWORD);
        const int groupIndex =
            static_cast<int>(logicalBlock % GROUPS_PER_PASSWORD);
        const int guessIndex = groupIndex * blockDim.x + threadIdx.x;

        if (guessIndex >= TOTAL_CANDIDATES) {
            continue;
        }

        // Convert guess index to a candidate in the range aa00 to zz99.
        int x = guessIndex;
        char raw[RAW_STRIDE];
        raw[3] = static_cast<char>('0' + x % 10);
        x /= 10;
        raw[2] = static_cast<char>('0' + x % 10);
        x /= 10;
        raw[1] = static_cast<char>('a' + x % 26);
        x /= 26;
        raw[0] = static_cast<char>('a' + x);
        raw[RAW_LENGTH] = '\0';

        char encryptedCandidate[ENCRYPTED_STRIDE];
        cudaCryptDevice(raw, encryptedCandidate);

        const char *target = encryptedInput + passwordIndex * ENCRYPTED_STRIDE;
        bool matched = true;
        for (int k = 0; k < ENCRYPTED_LENGTH; ++k) {
            if (encryptedCandidate[k] != target[k]) {
                matched = false;
                break;
            }
        }

        // Atomic CAS guarantees only one matching thread writes a password.
        if (matched && atomicCAS(&foundFlags[passwordIndex], 0, 1) == 0) {
            char *destination = recoveredOutput + passwordIndex * RAW_STRIDE;
            for (int j = 0; j < RAW_LENGTH; ++j) {
                destination[j] = raw[j];
            }
            destination[RAW_LENGTH] = '\0';
        }
    }
}

static bool cudaOK(cudaError_t error, const char *operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n",
                operation, cudaGetErrorString(error));
        return false;
    }
    return true;
}

static bool readEncryptedFile(const char *filename,
                              std::vector<char> &encrypted,
                              int &passwordCount) {
    FILE *input = fopen(filename, "r");
    if (input == NULL) {
        perror("Cannot open encrypted-password input file");
        return false;
    }

    char line[4096];
    int lineNumber = 0;
    bool success = true;

    while (fgets(line, sizeof(line), input) != NULL) {
        ++lineNumber;
        size_t length = strlen(line);
        while (length > 0 && (line[length - 1] == '\n' ||
                              line[length - 1] == '\r')) {
            line[--length] = '\0';
        }

        if (length != ENCRYPTED_LENGTH) {
            fprintf(stderr, "Invalid encrypted password on line %d: "
                            "expected exactly 10 characters, got %zu.\n",
                    lineNumber, length);
            success = false;
            break;
        }
        for (int j = 0; j < ENCRYPTED_LENGTH; ++j) {
            bool expectedCharacter = (j < 6)
                ? (line[j] >= 'a' && line[j] <= 'z')
                : (line[j] >= '0' && line[j] <= '9');
            if (!expectedCharacter) {
                fprintf(stderr, "Invalid character on line %d, column %d.\n",
                        lineNumber, j + 1);
                success = false;
                break;
            }
        }
        if (!success) break;

        encrypted.insert(encrypted.end(), line, line + ENCRYPTED_LENGTH);
        encrypted.push_back('\0');
        if (lineNumber == INT_MAX) {
            fprintf(stderr, "Input contains too many passwords.\n");
            success = false;
            break;
        }
    }

    if (ferror(input)) {
        perror("Error reading encrypted-password file");
        success = false;
    }
    if (fclose(input) != 0) {
        perror("Error closing encrypted-password file");
        success = false;
    }
    if (success && lineNumber == 0) {
        fprintf(stderr, "Input file contains no passwords.\n");
        success = false;
    }

    if (success) passwordCount = lineNumber;
    return success;
}

int main(int argc, char **argv) {
    if (argc != 2 && argc != 3) {
        fprintf(stderr, "Usage: %s <encrypted_input.txt> [decrypted_output.txt]\n",
                argv[0]);
        return 1;
    }
    const char *outputPath = (argc == 3) ? argv[2] : "decrypted.txt";

    std::vector<char> hostEncrypted;
    int passwordCount = 0;
    if (!readEncryptedFile(argv[1], hostEncrypted, passwordCount)) return 1;

    const size_t encryptedBytes = hostEncrypted.size();
    const size_t recoveredBytes =
        static_cast<size_t>(passwordCount) * RAW_STRIDE;
    const size_t flagBytes = static_cast<size_t>(passwordCount) * sizeof(int);
    std::vector<char> hostRecovered(recoveredBytes, '\0');
    std::vector<int> hostFound(passwordCount, 0);

    char *deviceEncrypted = NULL;
    char *deviceRecovered = NULL;
    int *deviceFound = NULL;
    int exitStatus = 1;

    do {
        int deviceCount = 0;
        if (!cudaOK(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount")) break;
        if (deviceCount < 1) {
            fprintf(stderr, "No CUDA GPU is available.\n");
            break;
        }
        if (!cudaOK(cudaMalloc(reinterpret_cast<void **>(&deviceEncrypted),
                               encryptedBytes), "cudaMalloc encrypted")) break;
        if (!cudaOK(cudaMalloc(reinterpret_cast<void **>(&deviceRecovered),
                               recoveredBytes), "cudaMalloc recovered")) break;
        if (!cudaOK(cudaMalloc(reinterpret_cast<void **>(&deviceFound),
                               flagBytes), "cudaMalloc flags")) break;

        if (!cudaOK(cudaMemcpy(deviceEncrypted, hostEncrypted.data(),
                               encryptedBytes, cudaMemcpyHostToDevice),
                    "cudaMemcpy input host-to-device")) break;
        if (!cudaOK(cudaMemset(deviceRecovered, 0, recoveredBytes),
                    "cudaMemset recovered")) break;
        if (!cudaOK(cudaMemset(deviceFound, 0, flagBytes),
                    "cudaMemset flags")) break;

        const unsigned long long logicalBlocks =
            static_cast<unsigned long long>(passwordCount) * GROUPS_PER_PASSWORD;
        // Keep the physical grid bounded; logical work scales with file size.
        const int gridBlocks = static_cast<int>(
            logicalBlocks < 65535ULL ? logicalBlocks : 65535ULL);

        printf("Encrypted passwords loaded: %d\n", passwordCount);
        printf("Candidate passwords per encrypted item: %d\n", TOTAL_CANDIDATES);
        printf("CUDA launch: %d blocks x %d threads\n",
               gridBlocks, THREADS_PER_BLOCK);

        crackPasswords<<<gridBlocks, THREADS_PER_BLOCK>>>(
            deviceEncrypted, deviceRecovered, deviceFound, passwordCount);
        if (!cudaOK(cudaGetLastError(), "kernel launch")) break;
        if (!cudaOK(cudaDeviceSynchronize(), "kernel execution")) break;

        if (!cudaOK(cudaMemcpy(hostRecovered.data(), deviceRecovered,
                               recoveredBytes, cudaMemcpyDeviceToHost),
                    "cudaMemcpy recovered device-to-host")) break;
        if (!cudaOK(cudaMemcpy(hostFound.data(), deviceFound,
                               flagBytes, cudaMemcpyDeviceToHost),
                    "cudaMemcpy flags device-to-host")) break;

        int recoveredCount = 0;
        for (int i = 0; i < passwordCount; ++i) {
            if (hostFound[i]) ++recoveredCount;
        }
        printf("Passwords recovered: %d / %d\n",
               recoveredCount, passwordCount);

        FILE *output = fopen(outputPath, "w");
        if (output == NULL) {
            perror("Cannot open decrypted output file");
            break;
        }
        bool writeOK = true;
        for (int i = 0; i < passwordCount; ++i) {
            const char *raw = hostRecovered.data() + i * RAW_STRIDE;
            if (fprintf(output, "%s\n", hostFound[i] ? raw : "NOT_FOUND") < 0) {
                perror("Error writing decrypted password");
                writeOK = false;
                break;
            }
        }
        if (fclose(output) != 0) {
            perror("Error closing decrypted output file");
            writeOK = false;
        }
        if (!writeOK) break;
        printf("Results saved to %s\n", outputPath);
        exitStatus = (recoveredCount == passwordCount) ? 0 : 2;
    } while (false);

    // Free all GPU allocations, including on error paths.
    if (deviceEncrypted != NULL &&
        !cudaOK(cudaFree(deviceEncrypted), "cudaFree encrypted")) exitStatus = 1;
    if (deviceRecovered != NULL &&
        !cudaOK(cudaFree(deviceRecovered), "cudaFree recovered")) exitStatus = 1;
    if (deviceFound != NULL &&
        !cudaOK(cudaFree(deviceFound), "cudaFree flags")) exitStatus = 1;

    return exitStatus;
}
