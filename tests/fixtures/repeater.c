#include <stdio.h>
#include <stdlib.h>

int main()
{
    char *line = NULL;
    size_t len = 0;

    while (getline(&line, &len, stdin) != -1)
    {
        fputs(line, stdout);
        fflush(stdout);
    }
    free(line);
    return 0;
}
