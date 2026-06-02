#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

int main()
{
    float in[4];
    uint8_t flag = 0x01;

    while (fread(in, sizeof(float), 4, stdin) == 4) {
        fwrite(&flag, 1, 1, stdout);
        fwrite(in, sizeof(float), 3, stdout);
        fflush(stdout);
    }
    return 0;
}
