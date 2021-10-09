#!/bin/bash
set -e

javac -d build -bootclasspath $ANDROID_PLATFORM/android.jar -classpath src -source 1.7 -target 1.7 src/sk/linuxos/*.java
d8 build/sk/linuxos/*.class --release --output build/dex --no-desugaring
aapt package -f -F build/CalibrateCamera.apkPart -I $ANDROID_PLATFORM/android.jar -M AndroidManifest.xml -S res -v
CLASSPATH=$ANDROID_HOME/tools/lib/* java com.android.sdklib.build.ApkBuilderMain build/CalibrateCamera.apkUnalign -d -f build/dex/classes.dex -v -z build/CalibrateCamera.apkPart
zipalign -f -v 4 build/CalibrateCamera.apkUnalign build/CalibrateCamera.apk

adb install -r build/CalibrateCamera.apk
adb shell am start -n sk.linuxos/.CameraCalibrate

# adb forward tcp:8421 tcp:8421
# adb logcat sk.linuxos.CameraCalibrate:V *:S AndroidRuntime:E
# echo "setIso 100\nsetExposure 32809411\ngetPixelPattern\ngetRaw 0 0 0 0\nquit"|socat -,ignoreeof TCP4:127.0.0.1:8421
