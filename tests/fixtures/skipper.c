#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

int main()
{
    float in[7];
    uint8_t flag;
    int count = 0;

    while (fread(in, sizeof(float), 7, stdin) == 7) {
        if (count % 2 == 1) {
            flag = 0x01;
            fwrite(&flag, 1, 1, stdout);
            fwrite(in, sizeof(float), 6, stdout);
        } else {
            flag = 0x00;
            fwrite(&flag, 1, 1, stdout);
        }
        fflush(stdout);
        count++;
    }
    return 0;
}
