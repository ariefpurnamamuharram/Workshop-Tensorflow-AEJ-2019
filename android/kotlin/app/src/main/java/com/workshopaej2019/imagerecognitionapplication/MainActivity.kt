package com.workshopaej2019.imagerecognitionapplication

import android.graphics.Bitmap
import android.os.Bundle
import android.support.v4.content.ContextCompat
import android.support.v7.app.AppCompatActivity
import android.util.Log
import com.wonderkiln.camerakit.CameraKitImage
import com.workshopaej2019.imagerecognitionapplication.tf.Classifier
import com.workshopaej2019.imagerecognitionapplication.tf.TFImageClassifier
import kotlinx.android.synthetic.main.activity_main.*
import kotlinx.coroutines.experimental.Job
import kotlinx.coroutines.experimental.launch

class MainActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "MainActivity"
        private const val INPUT_SIZE = 224
    }

    private lateinit var modelPath: String
    private lateinit var labelPath: String

    private var classifier: Classifier? = null
    private var initializeJob: Job? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        initActivity()
        imgbtn_shutter.setOnClickListener {
            camera.captureImage {
                onImageCaptured(it)
            }
            shutterObject(true)
        }
    }

    override fun onStart() {
        super.onStart()
        camera.start()
    }

    override fun onPause() {
        super.onPause()
        camera.stop()
    }

    private fun initActivity() {
        // Define your model path here
        modelPath = "models/optimized_graph.lite"
        // Define your label path here
        labelPath = "models/retrained_labels.txt"
        // Initialize TensorFlow Classifier
        initializeTensorClassifier()
    }

    private fun initializeTensorClassifier() {
        initializeJob = launch {
            try {
                classifier = TFImageClassifier.create(
                    assets, modelPath, labelPath, INPUT_SIZE
                )
                runOnUiThread {
                    imgbtn_shutter.isEnabled = true
                }
            } catch (e: Exception) {
                throw RuntimeException("Error initializing TensorFlow!", e)
            }
        }
    }

    private fun onImageCaptured(it: CameraKitImage) {
        val bitmap = Bitmap.createScaledBitmap(it.bitmap, INPUT_SIZE, INPUT_SIZE, false)
        classifier?.let {
            try {
                showRecognizedResult(it.recognizeImage(bitmap))
            } catch (e: java.lang.RuntimeException) {
                Log.e(TAG, "Crashing due to classification.closed() before the recognizer finishes!")
            }
        }
    }

    private fun showRecognizedResult(results: MutableList<Classifier.Recognition>) {
        runOnUiThread {
            if (results.isEmpty()) {
                result.text = getString(R.string.nothing)
                confidentiality.text = getString(R.string.nothing)
            } else {
                val mResult = results[0].title
                val mConfidentiality = results[0].confidence.toString()
                result.text = mResult
                confidentiality.text = mConfidentiality
            }
            shutterObject(false)
        }
    }

    private fun shutterObject(obj: Boolean) {
        when (obj) {
            true -> {
                shutter_frame.setColorFilter(
                    ContextCompat.getColor(this, R.color.colorPrimaryDark),
                    android.graphics.PorterDuff.Mode.SRC_IN
                )
                shutter_frame.alpha = 0.5F
            }
            false -> {
                shutter_frame.setColorFilter(
                    ContextCompat.getColor(this, android.R.color.white),
                    android.graphics.PorterDuff.Mode.SRC_IN
                )
                shutter_frame.alpha = 0.5F
            }
        }
    }

}
