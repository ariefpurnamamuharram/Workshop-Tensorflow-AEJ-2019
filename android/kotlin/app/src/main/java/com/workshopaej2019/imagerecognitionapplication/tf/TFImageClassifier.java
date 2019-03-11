package com.workshopaej2019.imagerecognitionapplication.tf;

import android.annotation.SuppressLint;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.util.Log;
import org.tensorflow.lite.Interpreter;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;

/**
 * Created by amitshekhar on 17/03/18.
 */

public class TFImageClassifier implements Classifier {

    private static final String TAG = "TFImageClassifier";

    private static final int MAX_RESULTS = 3;
    private static final int BATCH_SIZE = 1;
    private static final int PIXEL_SIZE = 3;
    private static final float THRESHOLD = 0.1f;

    private static final int IMAGE_MEAN = 128;
    private static final float IMAGE_STD = 128.0f;

    private Interpreter interpreter;
    private int inputSize;
    private List<String> labelList;

    private TFImageClassifier() {
    }

    public static Classifier create(AssetManager assetManager,
                                    String modelPath,
                                    String labelPath,
                                    int inputSize) throws IOException {

        TFImageClassifier classifier = new TFImageClassifier();
        classifier.interpreter = new Interpreter(classifier.loadModelFile(assetManager, modelPath));
        classifier.labelList = classifier.loadLabelList(assetManager, labelPath);
        classifier.inputSize = inputSize;

        Log.d(TAG, "Classifier function OK!");

        return classifier;
    }

    @Override
    public List<Recognition> recognizeImage(Bitmap bitmap) {
        ByteBuffer byteBuffer = convertBitmapToByteBuffer(bitmap);
        float[][] result = new float[1][labelList.size()];
        interpreter.run(byteBuffer, result);

        Log.d(TAG, "recognizeImage function OK!");

        return getSortedResult(result);
    }

    @Override
    public void close() {
        interpreter.close();
        interpreter = null;

        Log.d(TAG, "close function OK!");
    }

    private MappedByteBuffer loadModelFile(AssetManager assetManager, String modelPath) throws IOException {
        AssetFileDescriptor fileDescriptor = assetManager.openFd(modelPath);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();

        Log.d(TAG, "MappedByteBuffer function OK!");

        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private List<String> loadLabelList(AssetManager assetManager, String labelPath) throws IOException {
        List<String> labelList = new ArrayList<>();
        BufferedReader reader = new BufferedReader(new InputStreamReader(assetManager.open(labelPath)));
        String line;
        while ((line = reader.readLine()) != null) {
            labelList.add(line);
        }
        reader.close();

        Log.d(TAG, "loadLabelList function OK!");

        return labelList;
    }

    private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * BATCH_SIZE * inputSize * inputSize * PIXEL_SIZE);
        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intValues = new int[inputSize * inputSize];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        int pixel = 0;
        for (int i = 0; i < inputSize; ++i) {
            for (int j = 0; j < inputSize; ++j) {
                final int val = intValues[pixel++];
                byteBuffer.putFloat((((val >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                byteBuffer.putFloat((((val >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                byteBuffer.putFloat((((val) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
            }
        }

        Log.d(TAG, "convertBitmapToByteBuffer function OK!");

        return byteBuffer;
    }

    @SuppressLint("DefaultLocale")
    private List<Recognition> getSortedResult(float[][] labelProbArray) {

        PriorityQueue<Recognition> pq =
                new PriorityQueue<>(
                        MAX_RESULTS,
                        new Comparator<Recognition>() {
                            @Override
                            public int compare(Recognition lhs, Recognition rhs) {
                                return Float.compare(rhs.getConfidence(), lhs.getConfidence());
                            }
                        });

        for (int i = 0; i < labelList.size(); ++i) {
            float confidence = labelProbArray[0][i];
            if (confidence > THRESHOLD) {
                pq.add(new Recognition("" + i,
                        labelList.size() > i ? labelList.get(i) : "unknown",
                        confidence));
            }
        }

        final ArrayList<Recognition> recognitions = new ArrayList<>();
        int recognitionsSize = Math.min(pq.size(), MAX_RESULTS);
        for (int i = 0; i < recognitionsSize; ++i) {
            recognitions.add(pq.poll());
        }

        Log.d(TAG, "getSortedResult function OK!");

        return recognitions;
    }

}
