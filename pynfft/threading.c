/* Threading utilities for pyNFFT
 * ------------------------------
 *
 * Note: The dummy wrappers for FFTW's fftw_init_threads() and
 *       fftw_cleanup_threads() are lifted from threads/api.c
 *       in FFTW's source code.
 */

#include "config.h"
/* #include "fftw3.h" */

#ifndef FFTW_THREADS
int fftw_init_threads(void) 
{ 
    return 0;
}
void fftw_cleanup_threads(void)
{
    return;
}
#endif  /* FFTW_THREADS */