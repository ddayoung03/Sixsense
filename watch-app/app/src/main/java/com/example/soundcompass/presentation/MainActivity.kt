package com.example.soundcompass.presentation

import android.Manifest
import android.annotation.SuppressLint
import android.bluetooth.*
import android.bluetooth.le.*
import android.content.Context
import android.content.SharedPreferences
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.*
import android.util.Log
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.BackHandler
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.NotificationsActive
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.filled.VolumeUp
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.wear.compose.material3.Icon
import androidx.wear.compose.material3.Text
import kotlinx.coroutines.delay
import java.text.SimpleDateFormat
import java.util.*

val SERVICE_UUID: UUID = UUID.fromString("4fafc201-1fb5-459e-8fcc-c5c9c331914b")
val CHAR_UUID: UUID = UUID.fromString("beb5483e-36e1-4688-b7f5-ea07361b26a8")

class MainActivity : ComponentActivity(), SensorEventListener {

    private var bluetoothGatt: BluetoothGatt? = null
    private var scanner: BluetoothLeScanner? = null
    private var wakeLock: PowerManager.WakeLock? = null

    private lateinit var sensorManager: SensorManager
    private var rotationSensor: Sensor? = null
    private val currentAzimuth = mutableStateOf(0f)
    private var initialAzimuth = 0f

    private val isConnected = mutableStateOf(false)
    private val isConnecting = mutableStateOf(false)
    private val isScanning = mutableStateOf(false)
    private var userInitiatedDisconnect = false
    private var reconnectAttempts = 0
    private val maxReconnectAttempts = 3

    private val scannedDevicesMap = mutableStateMapOf<String, BluetoothDevice>()
    private val deviceTimestamps = mutableMapOf<String, Long>()
    private val connectedDeviceName = mutableStateOf("")

    private val isAlertActive = mutableStateOf(false)
    private val alertSoundType = mutableStateOf("")
    private val alertAngle = mutableStateOf(0f)

    private val isSettingsOpen = mutableStateOf(false)
    private val clockColor = mutableStateOf(Color(0xFF76FF03))
    private val alertColor = mutableStateOf(Color(0xFFFF1744))
    private val vibeStrength = mutableStateOf(1)

    private lateinit var settingsPrefs: SharedPreferences

    private var savedDeviceAddress: String?
        get() = settingsPrefs.getString("last_device_address", null)
        set(value) { settingsPrefs.edit().putString("last_device_address", value).apply() }

    private var savedDeviceName: String?
        get() = settingsPrefs.getString("last_device_name", null)
        set(value) { settingsPrefs.edit().putString("last_device_name", value).apply() }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        settingsPrefs = getSharedPreferences("sixsense_settings", Context.MODE_PRIVATE)
        clockColor.value = Color(settingsPrefs.getInt("clock_color", clockColor.value.toArgb()))
        alertColor.value = Color(settingsPrefs.getInt("alert_color", alertColor.value.toArgb()))
        vibeStrength.value = settingsPrefs.getInt("vibe_strength", vibeStrength.value)

        setShowWhenLocked(true)
        setTurnScreenOn(true)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        val powerManager = getSystemService(Context.POWER_SERVICE) as PowerManager
        wakeLock = powerManager.newWakeLock(
            PowerManager.FULL_WAKE_LOCK or PowerManager.ACQUIRE_CAUSES_WAKEUP or PowerManager.ON_AFTER_RELEASE,
            "SoundCompass::AlertWakeLock"
        )

        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        rotationSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)

        val permissionLauncher = registerForActivityResult(
            ActivityResultContracts.RequestMultiplePermissions()
        ) { permissions ->
            if (!permissions.entries.all { it.value }) {
                Log.e("BLE", "권한 거부됨")
            } else {
                tryAutoReconnect()
            }
        }

        setContent {
            LaunchedEffect(Unit) {
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                    permissionLauncher.launch(arrayOf(
                        Manifest.permission.BLUETOOTH_SCAN,
                        Manifest.permission.BLUETOOTH_CONNECT,
                        Manifest.permission.ACCESS_FINE_LOCATION,
                        Manifest.permission.WAKE_LOCK
                    ))
                } else {
                    permissionLauncher.launch(arrayOf(Manifest.permission.ACCESS_FINE_LOCATION))
                }
            }

            LaunchedEffect(isScanning.value) {
                while (isScanning.value) {
                    delay(500)
                    val now = System.currentTimeMillis()
                    val disconnectedDevices = deviceTimestamps.filter { now - it.value > 1000 }.keys
                    disconnectedDevices.forEach { key ->
                        deviceTimestamps.remove(key)
                        scannedDevicesMap.remove(key)
                    }
                }
            }

            LaunchedEffect(isAlertActive.value, alertSoundType.value, alertAngle.value) {
                if (isAlertActive.value) {
                    // The board only notifies while it's actively re-confirming the
                    // sound (every ~0.25s), so a stopped stream of notifications
                    // itself means the sound has stopped - this timeout just needs
                    // to cover a couple of missed BLE packets, not act as the
                    // primary detector. Was 5000ms, which made the alert visibly
                    // linger after the sound actually ended.
                    delay(1500)
                    isAlertActive.value = false
                }
            }

            if (isConnected.value) {
                val rotationOffset = currentAzimuth.value - initialAzimuth
                val compensatedAngle = alertAngle.value - rotationOffset

                MainScreen(
                    isAlertActive = isAlertActive.value,
                    soundType = alertSoundType.value,
                    angle = compensatedAngle,
                    isSettingsOpen = isSettingsOpen.value,
                    connectedDeviceName = connectedDeviceName.value,
                    clockColor = clockColor.value,
                    alertColor = alertColor.value,
                    vibeStrength = vibeStrength.value,
                    onOpenSettings = { isSettingsOpen.value = true },
                    onCloseSettings = { isSettingsOpen.value = false },
                    onDisconnect = { disconnectFromDevice() },
                    onClockColorChange = {
                        clockColor.value = it
                        settingsPrefs.edit().putInt("clock_color", it.toArgb()).apply()
                    },
                    onAlertColorChange = {
                        alertColor.value = it
                        settingsPrefs.edit().putInt("alert_color", it.toArgb()).apply()
                    },
                    onVibeChange = {
                        vibeStrength.value = it
                        settingsPrefs.edit().putInt("vibe_strength", it).apply()
                        triggerHapticFeedback(this@MainActivity, it)
                    }
                )
            } else {
                ConnectScreen(
                    isScanning = isScanning.value,
                    isConnecting = isConnecting.value,
                    devices = scannedDevicesMap.values.toList(),
                    onStartScan = { startBleScan() },
                    onDeviceClick = { device -> connectToDevice(device) }
                )
            }
        }
    }

    override fun onResume() {
        super.onResume()
        rotationSensor?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME)
        }
    }

    override fun onPause() {
        super.onPause()
        sensorManager.unregisterListener(this)
    }

    override fun onSensorChanged(event: SensorEvent?) {
        if (event?.sensor?.type == Sensor.TYPE_ROTATION_VECTOR) {
            val rotationMatrix = FloatArray(9)
            SensorManager.getRotationMatrixFromVector(rotationMatrix, event.values)

            // getOrientation()'s azimuth is defined for a device lying flat
            // (screen up); a watch is actually viewed tilted toward the wearer's
            // face, so the raw azimuth is off by however much the wrist is
            // raised. Remap to the "held upright, screen facing the wearer"
            // frame (the standard compass-app mapping) before reading azimuth,
            // per https://developer.android.com/develop/sensors-and-location/sensors/sensors_position.
            val remappedMatrix = FloatArray(9)
            SensorManager.remapCoordinateSystem(
                rotationMatrix, SensorManager.AXIS_X, SensorManager.AXIS_Z, remappedMatrix
            )

            val orientationAngles = FloatArray(3)
            SensorManager.getOrientation(remappedMatrix, orientationAngles)

            var azimuth = Math.toDegrees(orientationAngles[0].toDouble()).toFloat()
            if (azimuth < 0) azimuth += 360f

            // Low-pass filter over the shortest angular path so raw sensor jitter
            // doesn't show up as small steps in the direction indicator.
            val shortestDelta = ((azimuth - currentAzimuth.value + 540f) % 360f) - 180f
            currentAzimuth.value = (currentAzimuth.value + shortestDelta * 0.35f + 360f) % 360f
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    @SuppressLint("MissingPermission")
    private fun startBleScan() {
        try {
            val bluetoothManager = getSystemService(Context.BLUETOOTH_SERVICE) as BluetoothManager
            val bluetoothAdapter = bluetoothManager.adapter

            if (bluetoothAdapter == null || !bluetoothAdapter.isEnabled) return

            scanner = bluetoothAdapter.bluetoothLeScanner
            if (scanner == null) return

            if (isScanning.value) {
                scanner?.stopScan(scanCallback)
            }

            val scanSettings = ScanSettings.Builder().setScanMode(ScanSettings.SCAN_MODE_LOW_LATENCY).build()

            scannedDevicesMap.clear()
            deviceTimestamps.clear()
            isScanning.value = true

            scanner?.startScan(null, scanSettings, scanCallback)
        } catch (e: Exception) {
            Log.e("BLE", "에러: ${e.message}")
        }
    }

    private val scanCallback = object : ScanCallback() {
        @SuppressLint("MissingPermission")
        override fun onScanResult(callbackType: Int, result: ScanResult) {
            val device = result.device
            val deviceName = device.name ?: ""

            if (deviceName.contains("ESP32")) {
                scannedDevicesMap[device.address] = device
                deviceTimestamps[device.address] = System.currentTimeMillis()
            }
        }
    }

    @SuppressLint("MissingPermission")
    private fun connectToDevice(device: BluetoothDevice) {
        isScanning.value = false
        isConnecting.value = true
        scanner?.stopScan(scanCallback)
        connectedDeviceName.value = device.name ?: savedDeviceName ?: "알 수 없는 기기"
        bluetoothGatt = device.connectGatt(this, false, gattCallback)
    }

    @SuppressLint("MissingPermission")
    private fun tryAutoReconnect() {
        val address = savedDeviceAddress ?: return
        if (reconnectAttempts >= maxReconnectAttempts) return
        reconnectAttempts++
        try {
            val bluetoothManager = getSystemService(Context.BLUETOOTH_SERVICE) as BluetoothManager
            val adapter = bluetoothManager.adapter
            if (adapter == null || !adapter.isEnabled) return
            connectToDevice(adapter.getRemoteDevice(address))
        } catch (e: Exception) {
            Log.e("BLE", "자동 재연결 실패: ${e.message}")
        }
    }

    @SuppressLint("MissingPermission")
    private fun disconnectFromDevice() {
        userInitiatedDisconnect = true
        savedDeviceAddress = null
        savedDeviceName = null
        bluetoothGatt?.disconnect()
        bluetoothGatt?.close()
        bluetoothGatt = null
        isConnected.value = false
        isConnecting.value = false
        isSettingsOpen.value = false
    }

    private val gattCallback = object : BluetoothGattCallback() {
        @SuppressLint("MissingPermission")
        override fun onConnectionStateChange(gatt: BluetoothGatt, status: Int, newState: Int) {
            if (newState == BluetoothProfile.STATE_CONNECTED) {
                gatt.discoverServices()
            } else if (newState == BluetoothProfile.STATE_DISCONNECTED) {
                gatt.close()
                isConnected.value = false
                if (userInitiatedDisconnect) {
                    userInitiatedDisconnect = false
                    isConnecting.value = false
                } else {
                    tryAutoReconnect()
                }
            }
        }

        @SuppressLint("MissingPermission")
        override fun onServicesDiscovered(gatt: BluetoothGatt, status: Int) {
            if (status == BluetoothGatt.GATT_SUCCESS) {
                val service = gatt.getService(SERVICE_UUID)
                val characteristic = service?.getCharacteristic(CHAR_UUID)

                if (characteristic != null) {
                    gatt.setCharacteristicNotification(characteristic, true)
                    val descriptor = characteristic.getDescriptor(UUID.fromString("00002902-0000-1000-8000-00805f9b34fb"))
                    if (descriptor != null) {
                        descriptor.value = BluetoothGattDescriptor.ENABLE_NOTIFICATION_VALUE
                        gatt.writeDescriptor(descriptor)
                        isConnected.value = true
                        isConnecting.value = false
                        reconnectAttempts = 0
                        savedDeviceAddress = gatt.device.address
                        gatt.device.name?.let { savedDeviceName = it }
                    }
                }
            }
        }

        override fun onCharacteristicChanged(gatt: BluetoothGatt, characteristic: BluetoothGattCharacteristic) {
            val receivedData = characteristic.getStringValue(0)
            val parts = receivedData?.split(",")

            if (parts != null && parts.size == 2) {
                wakeUpScreen()

                alertSoundType.value = parts[0].trim().uppercase()
                alertAngle.value = parts[1].trim().toFloatOrNull() ?: 0f
                initialAzimuth = currentAzimuth.value

                isSettingsOpen.value = false
                isAlertActive.value = true
                triggerHapticFeedback(this@MainActivity, vibeStrength.value, alertSoundType.value)
            }
        }
    }

    private fun wakeUpScreen() {
        if (wakeLock?.isHeld == false) {
            wakeLock?.acquire(3000)
        }
    }

    private fun triggerHapticFeedback(context: Context, level: Int, soundType: String? = null) {
        val vibratorManager = context.getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as VibratorManager
        val vibrator = vibratorManager.defaultVibrator

        // Rhythm depends on the danger type: horn = short/weak single pulse,
        // siren = strong/repeating pulses. Without a soundType (settings preview),
        // fall back to the old strength-only pattern.
        val pattern = when (soundType) {
            "HORN" -> longArrayOf(0, 60)
            "SIREN" -> longArrayOf(0, 200, 100, 200, 100, 200, 100, 200)
            else -> when (level) {
                0 -> longArrayOf(0, 40)
                1 -> longArrayOf(0, 80, 80, 80)
                2 -> longArrayOf(0, 150, 80, 150)
                else -> longArrayOf(0, 100, 40, 100, 40, 100)
            }
        }

        // Strength setting (약/중/강/최상) scales amplitude on top of the rhythm.
        val amplitude = when (level) {
            0 -> 90
            1 -> 150
            2 -> 200
            else -> 255
        }
        val amplitudes = IntArray(pattern.size) { i -> if (i % 2 == 0) 0 else amplitude }

        val effect = VibrationEffect.createWaveform(pattern, amplitudes, -1)
        vibrator.vibrate(effect)
    }

    @SuppressLint("MissingPermission")
    override fun onDestroy() {
        super.onDestroy()
        scanner?.stopScan(scanCallback)
        bluetoothGatt?.close()
        if (wakeLock?.isHeld == true) {
            wakeLock?.release()
        }
    }
}

// ---------------- UI 영역 ----------------

@SuppressLint("MissingPermission")
@Composable
fun ConnectScreen(
    isScanning: Boolean,
    isConnecting: Boolean,
    devices: List<BluetoothDevice>,
    onStartScan: () -> Unit,
    onDeviceClick: (BluetoothDevice) -> Unit
) {
    Column(
        modifier = Modifier.fillMaxSize().background(Color.Black).padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        if (isConnecting) {
            Text("이전 기기에 재연결 중...", color = Color.White, fontSize = 14.sp)
        } else if (!isScanning && devices.isEmpty()) {
            Box(
                modifier = Modifier.background(Color.DarkGray, RoundedCornerShape(20.dp)).clickable { onStartScan() }.padding(16.dp)
            ) { Text("기기 검색 시작", color = Color.White, fontSize = 16.sp) }
        } else {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.Center,
                modifier = Modifier.padding(bottom = 12.dp)
            ) {
                Text(
                    text = if (devices.isEmpty()) "보드를 찾는 중..." else "기기 선택",
                    color = Color.White,
                    fontSize = 14.sp
                )
                Spacer(modifier = Modifier.width(8.dp))
                Box(
                    modifier = Modifier
                        .size(24.dp)
                        .clip(CircleShape)
                        .background(Color(0xFF333333))
                        .clickable { onStartScan() },
                    contentAlignment = Alignment.Center
                ) {
                    Icon(
                        imageVector = Icons.Default.Refresh,
                        contentDescription = "새로고침",
                        tint = Color.White,
                        modifier = Modifier.size(16.dp)
                    )
                }
            }

            if (devices.isNotEmpty()) {
                LazyColumn(modifier = Modifier.fillMaxWidth(), horizontalAlignment = Alignment.CenterHorizontally) {
                    items(devices) { device ->
                        Box(
                            modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp).background(Color(0xFF004D40), RoundedCornerShape(12.dp)).clickable { onDeviceClick(device) }.padding(12.dp),
                            contentAlignment = Alignment.Center
                        ) { Text(text = device.name ?: "알 수 없는 기기", color = Color.White, fontWeight = FontWeight.Bold, textAlign = TextAlign.Center) }
                    }
                }
            }
        }
    }
}

@Composable
fun MainScreen(
    isAlertActive: Boolean,
    soundType: String,
    angle: Float,
    isSettingsOpen: Boolean,
    connectedDeviceName: String,
    clockColor: Color,
    alertColor: Color,
    vibeStrength: Int,
    onOpenSettings: () -> Unit,
    onCloseSettings: () -> Unit,
    onDisconnect: () -> Unit,
    onClockColorChange: (Color) -> Unit,
    onAlertColorChange: (Color) -> Unit,
    onVibeChange: (Int) -> Unit
) {
    if (isAlertActive) {
        AlertScreen(soundType = soundType, angle = angle, themeColor = alertColor)
    } else if (isSettingsOpen) {
        SettingsScreen(
            connectedDeviceName = connectedDeviceName,
            clockColor = clockColor,
            alertColor = alertColor,
            vibeStrength = vibeStrength,
            onClose = onCloseSettings,
            onDisconnect = onDisconnect,
            onClockColorChange = onClockColorChange,
            onAlertColorChange = onAlertColorChange,
            onVibeChange = onVibeChange
        )
    } else {
        DigitalClockScreen(clockColor = clockColor, onOpenSettings = onOpenSettings)
    }
}

@Composable
fun DigitalClockScreen(clockColor: Color, onOpenSettings: () -> Unit) {
    var currentTime by remember { mutableStateOf("") }

    LaunchedEffect(Unit) {
        while (true) {
            val sdf = SimpleDateFormat("HH:mm:ss", Locale.getDefault())
            currentTime = sdf.format(Date())
            delay(1000)
        }
    }

    Box(
        modifier = Modifier.fillMaxSize().background(Color.Black),
        contentAlignment = Alignment.Center
    ) {
        Box(
            modifier = Modifier
                .align(Alignment.TopEnd)
                .padding(top = 28.dp, end = 32.dp)
                .size(32.dp)
                .clip(CircleShape)
                .background(Color.DarkGray)
                .clickable { onOpenSettings() },
            contentAlignment = Alignment.Center
        ) {
            Icon(
                imageVector = Icons.Default.Settings,
                contentDescription = "설정",
                tint = Color.White,
                modifier = Modifier.size(18.dp)
            )
        }

        Text(
            text = currentTime,
            color = clockColor,
            fontSize = 36.sp,
            fontWeight = FontWeight.Bold
        )
    }
}

@Composable
fun AlertScreen(soundType: String, angle: Float, themeColor: Color) {
    val cleanSoundType = soundType.trim().uppercase()
    val alertIcon = if (cleanSoundType == "SIREN") Icons.Default.NotificationsActive else Icons.Default.VolumeUp

    // Animate toward the new angle along the shortest path instead of snapping,
    // so both new BLE readings and watch rotation feel like continuous motion.
    val animatedAngle = remember { Animatable(angle) }
    LaunchedEffect(angle) {
        val shortestDelta = ((angle - animatedAngle.value + 540f) % 360f) - 180f
        animatedAngle.animateTo(
            animatedAngle.value + shortestDelta,
            animationSpec = tween(durationMillis = 150, easing = LinearEasing)
        )
    }

    Box(modifier = Modifier.fillMaxSize().background(Color.Black), contentAlignment = Alignment.Center) {
        Canvas(modifier = Modifier.fillMaxSize()) {
            val strokeWidth = 25f
            drawCircle(color = Color.DarkGray, style = Stroke(width = strokeWidth))
            drawArc(
                color = themeColor,
                startAngle = animatedAngle.value - 90f - 30f,
                sweepAngle = 60f,
                useCenter = false,
                style = Stroke(width = strokeWidth)
            )
        }

        Icon(
            imageVector = alertIcon,
            contentDescription = cleanSoundType,
            tint = themeColor,
            modifier = Modifier.size(72.dp)
        )
    }
}

enum class SettingsPage { MAIN, CLOCK_COLOR, ALERT_COLOR }

@Composable
fun SettingsScreen(
    connectedDeviceName: String,
    clockColor: Color,
    alertColor: Color,
    vibeStrength: Int,
    onClose: () -> Unit,
    onDisconnect: () -> Unit,
    onClockColorChange: (Color) -> Unit,
    onAlertColorChange: (Color) -> Unit,
    onVibeChange: (Int) -> Unit
) {
    var currentPage by remember { mutableStateOf(SettingsPage.MAIN) }

    BackHandler {
        if (currentPage != SettingsPage.MAIN) {
            currentPage = SettingsPage.MAIN
        } else {
            onClose()
        }
    }

    when (currentPage) {
        SettingsPage.MAIN -> {
            val vibeLabels = listOf("약", "중", "강", "최상")

            LazyColumn(
                modifier = Modifier
                    .fillMaxSize()
                    .background(Color.Black)
                    .padding(horizontal = 24.dp, vertical = 20.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                item {
                    Text("⚙️ 설정", color = Color.White, fontSize = 15.sp, fontWeight = FontWeight.Bold)
                    Spacer(modifier = Modifier.height(12.dp))
                }

                item {
                    Text("연결됨: $connectedDeviceName", color = Color.Gray, fontSize = 10.sp)
                    Spacer(modifier = Modifier.height(4.dp))
                    Box(
                        modifier = Modifier
                            .fillMaxWidth(0.8f)
                            .clip(RoundedCornerShape(8.dp))
                            .background(Color(0xFFFF1744))
                            .clickable { onDisconnect() }
                            .padding(vertical = 6.dp),
                        contentAlignment = Alignment.Center
                    ) {
                        Text("기기 변경하기", color = Color.White, fontSize = 10.sp, fontWeight = FontWeight.Bold)
                    }
                    Spacer(modifier = Modifier.height(16.dp))
                }

                item {
                    ColorSettingButton(
                        label = "시계 색상 변경",
                        currentColor = clockColor,
                        onClick = { currentPage = SettingsPage.CLOCK_COLOR }
                    )
                    Spacer(modifier = Modifier.height(12.dp))
                }

                item {
                    ColorSettingButton(
                        label = "경고 색상 변경",
                        currentColor = alertColor,
                        onClick = { currentPage = SettingsPage.ALERT_COLOR }
                    )
                    Spacer(modifier = Modifier.height(16.dp))
                }

                item {
                    Text("진동 세기: ${vibeLabels[vibeStrength]}", color = Color.Gray, fontSize = 11.sp)
                    Spacer(modifier = Modifier.height(6.dp))
                    Row(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                        vibeLabels.forEachIndexed { index, label ->
                            Box(
                                modifier = Modifier
                                    .weight(1f)
                                    .clip(RoundedCornerShape(8.dp))
                                    .background(if (vibeStrength == index) Color(0xFF00E5FF) else Color.DarkGray)
                                    .clickable { onVibeChange(index) }
                                    .padding(vertical = 6.dp),
                                contentAlignment = Alignment.Center
                            ) {
                                Text(
                                    text = label,
                                    color = if (vibeStrength == index) Color.Black else Color.White,
                                    fontSize = 10.sp,
                                    fontWeight = FontWeight.Bold
                                )
                            }
                        }
                    }
                    Spacer(modifier = Modifier.height(18.dp))
                }

                item {
                    Box(
                        modifier = Modifier
                            .fillMaxWidth(0.8f)
                            .background(Color.DarkGray, RoundedCornerShape(12.dp))
                            .clickable { onClose() }
                            .padding(vertical = 6.dp),
                        contentAlignment = Alignment.Center
                    ) {
                        Text("완료", color = Color.White, fontSize = 11.sp, fontWeight = FontWeight.Bold)
                    }
                }
            }
        }

        SettingsPage.CLOCK_COLOR -> {
            ColorPickerSubScreen(
                title = "시계 색상",
                currentColor = clockColor,
                onColorSelected = onClockColorChange,
                onApply = { currentPage = SettingsPage.MAIN }
            )
        }

        SettingsPage.ALERT_COLOR -> {
            ColorPickerSubScreen(
                title = "경고 색상",
                currentColor = alertColor,
                onColorSelected = onAlertColorChange,
                onApply = { currentPage = SettingsPage.MAIN }
            )
        }
    }
}

@Composable
fun ColorSettingButton(label: String, currentColor: Color, onClick: () -> Unit) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(12.dp))
            .background(Color(0xFF222222))
            .clickable { onClick() }
            .padding(horizontal = 16.dp, vertical = 12.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Text(label, color = Color.White, fontSize = 12.sp, fontWeight = FontWeight.Bold)
        Box(
            modifier = Modifier
                .size(20.dp)
                .clip(CircleShape)
                .background(currentColor)
        )
    }
}

// 💡 둥근 화면 맞춤형으로 꽉 채운 쾌적한 색상 조절 화면
@Composable
fun ColorPickerSubScreen(
    title: String,
    currentColor: Color,
    onColorSelected: (Color) -> Unit,
    onApply: () -> Unit
) {
    val scrollState = rememberScrollState()

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.Black)
            .verticalScroll(scrollState) // 넘칠 경우 스크롤로 안전하게 터치 가능
            .padding(vertical = 24.dp, horizontal = 12.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Text(title, color = Color.White, fontSize = 14.sp, fontWeight = FontWeight.Bold)
        Spacer(modifier = Modifier.height(10.dp))

        // 1. 축소된 미리보기 (공간 절약)
        Box(
            modifier = Modifier
                .size(36.dp)
                .clip(CircleShape)
                .background(currentColor)
        )
        Spacer(modifier = Modifier.height(16.dp))

        // 2. 극대화된 색상 슬라이더 (높이 48dp, 둥근 베젤에 잘리지 않도록 너비 90%)
        val gradientColors = listOf(
            Color(0xFFFF1744), Color(0xFFFFEA00), Color(0xFF76FF03),
            Color(0xFF00E5FF), Color(0xFFD500F9), Color(0xFFFFFFFF)
        )
        Box(
            modifier = Modifier
                .fillMaxWidth(0.9f)
                .height(48.dp)
                .clip(RoundedCornerShape(24.dp)) // 양끝을 동그랗게
                .background(Brush.horizontalGradient(gradientColors))
                .pointerInput(Unit) {
                    detectTapGestures { offset ->
                        val fraction = (offset.x / size.width).coerceIn(0f, 1f)
                        onColorSelected(getGradientColorAt(gradientColors, fraction))
                    }
                }
                .pointerInput(Unit) {
                    detectDragGestures { change, _ ->
                        change.consume()
                        val fraction = (change.position.x / size.width).coerceIn(0f, 1f)
                        onColorSelected(getGradientColorAt(gradientColors, fraction))
                    }
                }
        )
        Spacer(modifier = Modifier.height(20.dp))

        // 3. 적용 버튼
        Box(
            modifier = Modifier
                .fillMaxWidth(0.8f)
                .clip(RoundedCornerShape(12.dp))
                .background(Color(0xFF00E5FF))
                .clickable { onApply() }
                .padding(vertical = 10.dp),
            contentAlignment = Alignment.Center
        ) {
            Text("적용하기", color = Color.Black, fontSize = 14.sp, fontWeight = FontWeight.Bold)
        }
    }
}

fun getGradientColorAt(colors: List<Color>, fraction: Float): Color {
    if (fraction <= 0f) return colors.first()
    if (fraction >= 1f) return colors.last()

    val step = 1f / (colors.size - 1)
    val index = (fraction / step).toInt()
    val localFraction = (fraction - (index * step)) / step

    val startColor = colors[index]
    val endColor = colors[index + 1]

    return Color(
        red = startColor.red + localFraction * (endColor.red - startColor.red),
        green = startColor.green + localFraction * (endColor.green - startColor.green),
        blue = startColor.blue + localFraction * (endColor.blue - startColor.blue),
        alpha = 1f
    )
}