package com.example.alexnet;

public class netlib {
    static {
        System.loadLibrary("classify-lib");
    }

    public native String stringFromJNI();
    public native long initEnv(String model,String param,String label, float[] mean,int mean_size, int input_size);
    public native String inference(long env, String image,int limit);
}
