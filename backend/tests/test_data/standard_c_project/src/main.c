#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_BUFFER 1024

int init_system(void) {
    printf("System initialized\n");
    return 0;
}

int process_data(const char *buf, int len) {
    if (!buf || len <= 0) {
        return -1;
    }
    printf("Processed %d bytes\n", len);
    return 0;
}

void shutdown_system(void) {
    printf("System shutdown complete\n");
}

int main(int argc, char **argv) {
    init_system();
    char data[MAX_BUFFER];
    memset(data, 0, sizeof(data));
    process_data(data, 100);
    shutdown_system();
    return 0;
}
