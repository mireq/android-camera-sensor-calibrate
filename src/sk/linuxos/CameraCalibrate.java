package sk.linuxos;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.lang.Integer;
import java.lang.InterruptedException;
import java.lang.Long;
import java.lang.Runnable;
import java.lang.Thread;
import java.net.ServerSocket;
import java.net.Socket;
import java.net.SocketException;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Semaphore;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.ImageFormat;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.CaptureResult;
import android.hardware.camera2.TotalCaptureResult;
import android.hardware.camera2.params.OutputConfiguration;
import android.hardware.camera2.params.SessionConfiguration;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.hardware.camera2.params.RggbChannelVector;
import android.hardware.camera2.params.BlackLevelPattern;
import android.media.Image;
import android.media.ImageReader;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.util.Range;
import android.util.Size;
import android.widget.TextView;


public final class CameraCalibrate extends android.app.Activity
{
	private static final String TAG = "sk.linuxos.CameraCalibrate";
	private static final int port = 8421;
	private CameraDevice cameraDevice;
	private ImageReader imageReader;
	private Size rawSize;
	private CameraCaptureSession captureSession;
	private Thread serverThread = null;
	private ServerSocket serverSocket = null;
	private byte[] imageData;
	private Semaphore imageDataSemaphore;
	private Range<Integer> isoRange;
	private Range<Long> exposureRange;
	private int iso;
	private long exposure;
	private int capX = 0;
	private int capY = 0;
	private int capW = 0;
	private int capH = 0;
	private int capCount = 1;
	private int capCurrent = 0;
	private int imageCurrent = 0;
	private static final int bytesPerPixel = 2;
	private CaptureResult lastCaptureResult = null;
	private BlackLevelPattern blackLevelPattern = null;
	private String pixelPattern = null;

	@Override
	protected void onCreate(final android.os.Bundle activityState)
	{
		super.onCreate(activityState);
		final TextView textView = new TextView(CameraCalibrate.this);
		textView.setText("Listening on port " + port);
		setContentView(textView);
	}

	@Override
	protected void onResume()
	{
		super.onResume();
		openCamera();
	}

	@Override
	protected void onPause()
	{
		super.onPause();
		closeCamera();
	}

	final private CameraDevice.StateCallback stateCallback = new CameraDevice.StateCallback() {
		@Override
		public void onOpened(CameraDevice cdev) {
			cameraDevice = cdev;
			Log.v(TAG, "Camera opened");
		}

		@Override
		public void onDisconnected(CameraDevice cdev) {
			CameraCalibrate.this.closeCamera();
			cameraDevice = null;
		}

		@Override
		public void onError(CameraDevice cdev, int i) {
			Log.v(TAG, "Camera error");
			CameraCalibrate.this.closeCamera();
			CameraCalibrate.this.finish();
		}
	};

	final private CameraCaptureSession.StateCallback sessionStateCallback = new CameraCaptureSession.StateCallback()
	{
		@Override
		public void onConfigured(CameraCaptureSession session)
		{
			CameraCalibrate.this.captureSession = session;
			createCaptureRequest();
		}

		@Override
		public void onConfigureFailed(CameraCaptureSession session) {
			Log.e(TAG, "Camera configure failed");
			CameraCalibrate.this.finish();
		}
	};

	final private CameraCaptureSession.CaptureCallback captureSessionCallback = new CameraCaptureSession.CaptureCallback()
	{
		@Override
		public void onCaptureCompleted(CameraCaptureSession session, CaptureRequest request, TotalCaptureResult result) {
			Log.v(TAG, "Capture completed");
			lastCaptureResult = result;
			capCurrent++;
			if (imageCurrent >= capCount && capCurrent >= capCount) {
				imageDataSemaphore.release();
			}
			super.onCaptureCompleted(session, request, result);
		}
	};

	final private ImageReader.OnImageAvailableListener onImageAvailableListener = new ImageReader.OnImageAvailableListener()
	{
		public void onImageAvailable(ImageReader reader)
		{
			Log.v(TAG, "Image available");
			Image image = reader.acquireLatestImage();
			ByteBuffer buffer = image.getPlanes()[0].getBuffer();

			int pos = (capX + (capY * rawSize.getWidth())) * bytesPerPixel;
			int end = (1 + (capX + capW - 1) + ((capY + capH - 1) * rawSize.getWidth())) * bytesPerPixel;
			int lineWidth = (capW * bytesPerPixel);
			int stride = (rawSize.getWidth() - capW) * bytesPerPixel;
			int dstOffset = imageCurrent * capW * capH * bytesPerPixel;
			imageCurrent++;
			while (pos < end) {
				buffer.position(pos);
				buffer.get(imageData, dstOffset, lineWidth);
				dstOffset += lineWidth;
				pos += (lineWidth + stride);
			}

			image.close();
			if (imageCurrent >= capCount && capCurrent >= capCount) {
				imageDataSemaphore.release();
			}
		}
	};

	private void openCamera()
	{
		try {
			CameraManager cameraManager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);
			String[] cameraIdList = cameraManager.getCameraIdList();
			String mainCamera = null;
			CameraCharacteristics mainCameraCharacteristics = null;
			for (String cameraId: cameraIdList) {
				CameraCharacteristics characteristics = cameraManager.getCameraCharacteristics(cameraId);
				if (characteristics.get(CameraCharacteristics.LENS_FACING) == CameraCharacteristics.LENS_FACING_BACK) {
					mainCamera = cameraId;
					mainCameraCharacteristics = characteristics;
				}
			}
			if (mainCamera == null || mainCameraCharacteristics == null) {
				Log.e(TAG, "Main camera not found");
				this.finish();
				return;
			}
			blackLevelPattern = mainCameraCharacteristics.get(CameraCharacteristics.SENSOR_BLACK_LEVEL_PATTERN);
			StreamConfigurationMap streamConfiguration = mainCameraCharacteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
			rawSize = streamConfiguration.getOutputSizes(ImageFormat.RAW_SENSOR)[0];
			Log.v(TAG, "RAW size: " + rawSize.getWidth() + "x" + rawSize.getHeight());
			imageDataSemaphore = new Semaphore(1);
			imageDataSemaphore.acquire();

			// Set listener
			cameraManager.openCamera(mainCamera, stateCallback, null);

			isoRange = mainCameraCharacteristics.get(CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE);
			exposureRange = mainCameraCharacteristics.get(CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE);
			int colorFilterArrangement = mainCameraCharacteristics.get(CameraCharacteristics.SENSOR_INFO_COLOR_FILTER_ARRANGEMENT);
			switch (colorFilterArrangement) {
				case CameraCharacteristics.SENSOR_INFO_COLOR_FILTER_ARRANGEMENT_RGGB:
					pixelPattern = "RGGB";
					break;
				case CameraCharacteristics.SENSOR_INFO_COLOR_FILTER_ARRANGEMENT_GRBG:
					pixelPattern = "GRBG";
					break;
				case CameraCharacteristics.SENSOR_INFO_COLOR_FILTER_ARRANGEMENT_GBRG:
					pixelPattern = "GBRG";
					break;
				case CameraCharacteristics.SENSOR_INFO_COLOR_FILTER_ARRANGEMENT_BGGR:
					pixelPattern = "BGGR";
					break;
				case CameraCharacteristics.SENSOR_INFO_COLOR_FILTER_ARRANGEMENT_RGB:
					pixelPattern = "RGB";
					break;
				case CameraCharacteristics.SENSOR_INFO_COLOR_FILTER_ARRANGEMENT_MONO:
					pixelPattern = "MONO";
					break;
				case CameraCharacteristics.SENSOR_INFO_COLOR_FILTER_ARRANGEMENT_NIR:
					pixelPattern = "NIR";
					break;
			}

			Log.v(TAG, "Supported ISO: " + isoRange.getLower() + "-" + isoRange.getUpper());
			Log.v(TAG, "Supported Exposure: " + exposureRange.getLower() + "-" + exposureRange.getUpper());

			iso = isoRange.getLower();
			exposure = exposureRange.getLower();

			serverThread = new Thread(new Runnable() {
				@Override
				public void run()
				{
					runServer();
				}
			});
			serverThread.start();

		}
		catch (Throwable t) {
			printException(t);
			this.finish();
		}
	}

	private void closeCamera()
	{
		if (cameraDevice != null) {
			cameraDevice.close();
			cameraDevice = null;
			if (serverThread != null) {
				if (serverSocket != null) {
					try {
						serverSocket.close();
					}
					catch (IOException e) {
					}
				}
				serverThread = null;
				Log.v(TAG, "Closed");
			}
		}
	}

	private void printException(Throwable t) {
		//Log.e(TAG, t.getMessage());
		//t.printStackTrace();
		Log.e(TAG, Log.getStackTraceString(t));
	}

	private void captureRaw() {
		try {
			imageReader = ImageReader.newInstance(rawSize.getWidth(), rawSize.getHeight(), ImageFormat.RAW_SENSOR, 1);
			imageReader.setOnImageAvailableListener(onImageAvailableListener, null);
			OutputConfiguration outputConfig = new OutputConfiguration(imageReader.getSurface());
			SessionConfiguration config = new SessionConfiguration(SessionConfiguration.SESSION_REGULAR, Arrays.asList(outputConfig), CameraCalibrate.this.getMainExecutor(), sessionStateCallback);
			cameraDevice.createCaptureSession(config);
			//cameraDevice.createCaptureSession(Arrays.asList(imageReader.getSurface()), sessionStateCallback, null);
		}
		catch (Throwable t) {
			printException(t);
			finish();
		}
	}

	private void createCaptureRequest()
	{
		try {
			Log.v(TAG, "Create capture request");
			CaptureRequest.Builder captureRequestBuilder = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_MANUAL);
			captureRequestBuilder.addTarget(imageReader.getSurface());
			captureRequestBuilder.set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_OFF);
			captureRequestBuilder.set(CaptureRequest.LENS_FOCUS_DISTANCE, 0.0f);
			captureRequestBuilder.set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_OFF);
			captureRequestBuilder.set(CaptureRequest.CONTROL_AWB_MODE, CaptureRequest.CONTROL_AWB_MODE_AUTO);
			captureRequestBuilder.set(CaptureRequest.SENSOR_EXPOSURE_TIME, exposure);
			captureRequestBuilder.set(CaptureRequest.SENSOR_SENSITIVITY, iso);
			List<CaptureRequest> requests = new ArrayList<CaptureRequest>();
			for (int i = 0; i < capCount; ++i) {
				requests.add(captureRequestBuilder.build());
			}
			captureSession.captureBurst(requests, captureSessionCallback, null);
			//captureSession.capture(request, captureSessionCallback, null);
			//captureSession.setRepeatingRequest(request, captureSessionCallback, null);
		}
		catch (Throwable t) {
			printException(t);
			finish();
		}
	}

	private void runServer()
	{
		try {
			serverSocket = new ServerSocket(port);
			while(!Thread.currentThread().isInterrupted()) {
				Socket s = serverSocket.accept();
				try {
					BufferedReader in = new BufferedReader(new InputStreamReader(s.getInputStream()));
					BufferedWriter out = new BufferedWriter(new OutputStreamWriter(s.getOutputStream()));
					DataOutputStream dOut = new DataOutputStream(s.getOutputStream());
					String command = null;
					while (true) {
						command = in.readLine();
						if (command == null) {
							break;
						}
						String[] args = command.split(" ");
						command = args[0];
						boolean stop = false;
						switch (command) {
							case "quit":
								stop = true;
								break;
							case "getResolution":
								out.write(rawSize.getWidth() + " " + rawSize.getHeight() + "\n");
								out.flush();
								break;
							case "getBytesPerPixel":
								out.write("" + bytesPerPixel + "\n");
								out.flush();
								break;
							case "getIsoRange":
								out.write(isoRange.getLower() + " " + isoRange.getUpper() + "\n");
								out.flush();
								break;
							case "getExposureRange":
								out.write(exposureRange.getLower() + " " + exposureRange.getUpper() + "\n");
								out.flush();
								break;
							case "getPixelPattern":
								out.write(pixelPattern + "\n");
								out.flush();
								break;
							case "getRaw":
								capX = 0;
								capY = 0;
								capW = rawSize.getWidth();
								capH = rawSize.getHeight();
								capCount = 1;

								if (args.length >= 5) {
									capX = Integer.parseInt(args[1]);
									capY = Integer.parseInt(args[2]);
									capW = Integer.parseInt(args[3]);
									capH = Integer.parseInt(args[4]);
								}
								if (args.length >= 6) {
									capCount = Integer.parseInt(args[5]);
								}
								capCurrent = 0;
								imageCurrent = 0;

								imageData = new byte[capW * capH * bytesPerPixel * capCount];

								new Handler(Looper.getMainLooper()).post(new Runnable() {
									@Override
									public void run()
									{
										CameraCalibrate.this.captureRaw();
									}
								});
								imageDataSemaphore.acquire();
								RggbChannelVector colorCorrectionGains = lastCaptureResult.get(CaptureResult.COLOR_CORRECTION_GAINS);
								out.write("color_correction_gains: " + colorCorrectionGains.getRed() + " " + colorCorrectionGains.getGreenEven() + " " + colorCorrectionGains.getGreenOdd() + " " + colorCorrectionGains.getBlue() + "\n");
								float[] dynamicBlackLevel = lastCaptureResult.get(CaptureResult.SENSOR_DYNAMIC_BLACK_LEVEL);
								if (dynamicBlackLevel == null) {
									out.write("black_level: " + blackLevelPattern.getOffsetForIndex(0, 0) + " " + blackLevelPattern.getOffsetForIndex(1, 0) + " " + blackLevelPattern.getOffsetForIndex(0, 1) + " " + blackLevelPattern.getOffsetForIndex(1, 1) + "\n");
								}
								else {
									out.write("black_level: " + dynamicBlackLevel[0] + " " + dynamicBlackLevel[1] + " " + dynamicBlackLevel[2] + " " + dynamicBlackLevel[3] + "\n");
								}
								int dynamicWhiteLevel = lastCaptureResult.get(CaptureResult.SENSOR_DYNAMIC_WHITE_LEVEL);
								out.write("white_level: " + dynamicWhiteLevel + "\n");
								out.write("data: " + capW * capH * bytesPerPixel * capCount + "\n");
								out.flush();
								dOut.write(imageData, 0, capW * capH * bytesPerPixel * capCount);
								dOut.flush();
								break;
							case "setIso":
								iso = Integer.parseInt(args[1]);
								break;
							case "setExposure":
								exposure = Long.parseLong(args[1]);
								break;
							default:
								Log.e(TAG, "Unknown command " + command);
								break;
						}
						if (stop) {
							break;
						}
					}
				}
				catch (SocketException e) {
				}
				finally {
					s.close();
				}
			}
		}
		catch (SocketException e) {
			printException(e);
			serverSocket = null;
		}
		catch (Throwable t) {
			printException(t);
			finish();
		}
		finally {
			imageDataSemaphore = new Semaphore(1);
			try {
				imageDataSemaphore.acquire();
			}
			catch (Throwable t) {
			}
		}
	}
}
