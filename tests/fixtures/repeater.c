#include <stdio.h>
#include <stdlib.h>

int main()
{
    float in[7];
    
    while (fread(in, sizeof(float), 7, stdin) == 7) {
        fwrite(in, sizeof(float), 6, stdout);
        fflush(stdout);
    }
    return 0;
}
