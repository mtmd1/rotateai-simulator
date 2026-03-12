#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

int main()
{
    float in[7];
    uint8_t flag = 0x01;

    while (fread(in, sizeof(float), 7, stdin) == 7) {
        fwrite(&flag, 1, 1, stdout);
        fwrite(in, sizeof(float), 6, stdout);
        fflush(stdout);
    }
    return 0;
}
