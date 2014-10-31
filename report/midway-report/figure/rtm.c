// Example : Group sum operation in RTM
mutex fb_mutex;

#pragma omp parallel for 
for(int i = 0; i < N; i++){
    int mygroup = group[i];
    if(XBEGIN()) {  // First attempt
        if( !fb_mutex.is_acquired() ) {
            sums[mygroup] += data[i];
        }
        else{
            XABORT();
        }
        XEND();
    } else { // Fallback path: get mutex
        fb_mutex.acquire();
        sums[mygroup] += data[i];
        fb_mutex.release();
    }
}
