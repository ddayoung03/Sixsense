package com.example.sixsense.presentation

import android.Manifest
import com.example.sixsense.presentation.theme.SixSenseTheme
import android.annotation.SuppressLint
import android.bluetooth.*
import android.bluetooth.le.*
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
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
import androidx.core.content.edit
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
import androidx.compose.material.icons.automirrored.filled.VolumeUp
import androidx.compose.material.icons.filled.NotificationsActive
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.filled.BatteryFull
import androidx.compose.material.icons.filled.BatteryAlert
import androidx.compose.material.icons.filled.BatteryChargingFull
import androidx.compose.material.icons.filled.Battery1Bar
import androidx.compose.material.icons.filled.Battery2Bar
import androidx.compose.material.icons.filled.Battery3Bar
import androidx.compose.material.icons.filled.Battery4Bar
import androidx.compose.material.icons.filled.Battery5Bar
import androidx.compose.material.icons.filled.Battery6Bar
import androidx.compose.material.icons.filled.Battery0Bar
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
import kotlin.time.Duration.Companion.milliseconds

val SERVICE_UUID: UUID = UUID.fromString("4fafc201-1fb5-459e-8fcc-c5c9c331914b")
val CHAR_UUID: UUID = UUID.fromString("beb5483e-36e1-4688-b7f5-ea07361b26a8")

private val VIBRATE_PATTERNS = arrayOf(
    longArrayOf(0, 40),
    longArrayOf(0, 80, 80, 80),
    longArrayOf(0, 150, 80, 150),
    longArrayOf(0, 100, 40, 100, 40, 100)
)

class MainActivity : ComponentActivity(), SensorEventListener {

    private var bluetoothGatt: BluetoothGatt? = null
    private var scanner: BluetoothLeScanner? = null
    private var wakeLock: PowerManager.WakeLock? = null

    private lateinit var sensorManager: SensorManager
    private var rotationSensor: Sensor? = null
    private val currentAzimuth = mutableFloatStateOf(0f)
    private var initialAzimuth = 0f

    // 센서 연산을 위한 배열 재사용
    private val rotationMatrix = FloatArray(9)
    private val orientationAngles = FloatArray(3)

    private val isConnected = mutableStateOf(false)
    private val isScanning = mutableStateOf(false)

    private val scannedDevicesMap = mutableStateMapOf<String, BluetoothDevice>()
    private val deviceTimestamps = mutableMapOf<String, Long>()
    private val connectedDeviceName = mutableStateOf("")

    private val isAlertActive = mutableStateOf(false)
    private val alertSoundType = mutableStateOf("")
    private val alertAngle = mutableFloatStateOf(0f)

    private val esp32BatteryLevel = mutableIntStateOf(-1)
    private val isEsp32Charging = mutableStateOf(false)

    private val isSettingsOpen = mutableStateOf(false)
    private val clockColor = mutableStateOf(Color(0xFF76FF03))
    private val alertColor = mutableStateOf(Color(0xFFFF1744))
    private val vibeStrength = mutableIntStateOf(1)

    private val prefs by lazy { getSharedPreferences("SixSensePrefs", MODE_PRIVATE) }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // 설정 로드
        clockColor.value = Color(prefs.getInt("clockColor", Color(0xFF76FF03).toArgb()))
        alertColor.value = Color(prefs.getInt("alertColor", Color(0xFFFF1744).toArgb()))
        vibeStrength.intValue = prefs.getInt("vibeStrength", 1)

        setShowWhenLocked(true)
        setTurnScreenOn(true)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        val powerManager = getSystemService(POWER_SERVICE) as PowerManager
        @Suppress("DEPRECATION")
        wakeLock = powerManager.newWakeLock(
            PowerManager.FULL_WAKE_LOCK or PowerManager.ACQUIRE_CAUSES_WAKEUP or PowerManager.ON_AFTER_RELEASE,
            "SixSense::AlertWakeLock"
        )

        val bluetoothManager = getSystemService(BLUETOOTH_SERVICE) as BluetoothManager
        val bluetoothAdapter = bluetoothManager.adapter

        val enableBtLauncher = registerForActivityResult(
            ActivityResultContracts.StartActivityForResult()
        ) { result ->
            if (result.resultCode == RESULT_OK) {
                startBleScan()
            }
        }

        sensorManager = getSystemService(SENSOR_SERVICE) as SensorManager
        rotationSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)

        val permissionLauncher = registerForActivityResult(
            ActivityResultContracts.RequestMultiplePermissions()
        ) { permissions ->
            if (!permissions.entries.all { it.value }) {
                Log.e("BLE", "권한 거부됨")
            } else {
                // 권한 허용 후 자동 재연결 시도
                checkAutoReconnect()
            }
        }

        setContent {
            SixSenseTheme {
                LaunchedEffect(Unit) {
                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                        permissionLauncher.launch(
                            arrayOf(
                                Manifest.permission.BLUETOOTH_SCAN,
                                Manifest.permission.BLUETOOTH_CONNECT,
                                Manifest.permission.ACCESS_FINE_LOCATION,
                                Manifest.permission.WAKE_LOCK
                            )
                        )
                    } else {
                        permissionLauncher.launch(arrayOf(Manifest.permission.ACCESS_FINE_LOCATION))
                    }
                }

                LaunchedEffect(isScanning.value) {
                    while (isScanning.value) {
                        delay(500.milliseconds)
                        val now = System.currentTimeMillis()
                        val disconnectedDevices =
                            deviceTimestamps.filter { now - it.value > 1000 }.keys
                        disconnectedDevices.forEach { key ->
                            deviceTimestamps.remove(key)
                            scannedDevicesMap.remove(key)
                        }
                    }
                }

                LaunchedEffect(isAlertActive.value, alertSoundType.value, alertAngle.floatValue) {
                    if (isAlertActive.value) {
                        delay(5000.milliseconds)
                        isAlertActive.value = false
                    }
                }

                LaunchedEffect(isConnected.value) {
                    if (isConnected.value) {
                        requestBatteryUpdate() // 연결되자마자 즉시 한번 요청
                        while (isConnected.value) {
                            delay(60000.milliseconds) // 이후 1분마다
                            requestBatteryUpdate()
                        }
                    }
                }

                if (isConnected.value) {
                    val rotationOffset = currentAzimuth.floatValue - initialAzimuth
                    val compensatedAngle = alertAngle.floatValue - rotationOffset

                    MainScreen(
                        isAlertActive = isAlertActive.value,
                        soundType = alertSoundType.value,
                        angle = compensatedAngle,
                        isSettingsOpen = isSettingsOpen.value,
                        connectedDeviceName = connectedDeviceName.value,
                        clockColor = clockColor.value,
                        alertColor = alertColor.value,
                        vibeStrength = vibeStrength.intValue,
                        esp32BatteryLevel = esp32BatteryLevel.intValue,
                        isEsp32Charging = isEsp32Charging.value,
                        onOpenSettings = { isSettingsOpen.value = true },
                        onCloseSettings = { isSettingsOpen.value = false },
                        onDisconnect = { disconnectFromDevice() },
                        onClockColorChange = {
                            clockColor.value = it
                            prefs.edit { putInt("clockColor", it.toArgb()) }
                        },
                        onAlertColorChange = {
                            alertColor.value = it
                            prefs.edit { putInt("alertColor", it.toArgb()) }
                        },
                        onVibeChange = {
                            vibeStrength.intValue = it
                            prefs.edit { putInt("vibeStrength", it) }
                            triggerHapticFeedback(this@MainActivity, it)
                        }
                    )
                } else {
                    ConnectScreen(
                        isScanning = isScanning.value,
                        devices = scannedDevicesMap.values.toList(),
                        onStartScan = {
                            if (bluetoothAdapter?.isEnabled == false) {
                                enableBtLauncher.launch(Intent(BluetoothAdapter.ACTION_REQUEST_ENABLE))
                            } else {
                                startBleScan()
                            }
                        },
                        onDeviceClick = { device -> connectToDevice(device) }
                    )
                }
            }
        }
    }

    @SuppressLint("MissingPermission")
    private fun requestBatteryUpdate() {
        val service = bluetoothGatt?.getService(SERVICE_UUID)
        val characteristic = service?.getCharacteristic(CHAR_UUID)
        if (characteristic != null) {
            bluetoothGatt?.readCharacteristic(characteristic)
        }
    }

    private fun checkAutoReconnect() {
        val lastAddress = prefs.getString("lastDeviceAddress", null)
        if (lastAddress != null && !isConnected.value) {
            val bluetoothManager = getSystemService(BLUETOOTH_SERVICE) as BluetoothManager
            val device = bluetoothManager.adapter.getRemoteDevice(lastAddress)
            if (device != null) {
                connectedDeviceName.value = prefs.getString("lastDeviceName", "알 수 없는 기기") ?: "알 수 없는 기기"
                connectToDevice(device)
            }
        }
    }

    override fun onResume() {
        super.onResume()
        rotationSensor?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_UI)
        }
    }

    override fun onPause() {
        super.onPause()
        sensorManager.unregisterListener(this)
    }

    override fun onSensorChanged(event: SensorEvent?) {
        if (event?.sensor?.type == Sensor.TYPE_ROTATION_VECTOR) {
            SensorManager.getRotationMatrixFromVector(rotationMatrix, event.values)
            SensorManager.getOrientation(rotationMatrix, orientationAngles)

            var azimuth = Math.toDegrees(orientationAngles[0].toDouble()).toFloat()
            if (azimuth < 0) azimuth += 360f

            currentAzimuth.floatValue = azimuth
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    @SuppressLint("MissingPermission")
    private fun startBleScan() {
        try {
            val bluetoothManager = getSystemService(BLUETOOTH_SERVICE) as BluetoothManager
            val bluetoothAdapter = bluetoothManager.adapter

            if (bluetoothAdapter == null) return

            if (!bluetoothAdapter.isEnabled) {
                // 이 로직은 이제 ConnectScreen의 onStartScan에서 처리됨
                return
            }

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
        scanner?.stopScan(scanCallback)
        connectedDeviceName.value = device.name ?: "알 수 없는 기기"

        // 연결 정보 저장
        prefs.edit {
            putString("lastDeviceAddress", device.address)
            putString("lastDeviceName", device.name)
        }

        @Suppress("DEPRECATION")
        bluetoothGatt = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            device.connectGatt(this, false, gattCallback, BluetoothDevice.TRANSPORT_LE, BluetoothDevice.PHY_LE_1M_MASK, Handler(Looper.getMainLooper()))
        } else {
            device.connectGatt(this, false, gattCallback, BluetoothDevice.TRANSPORT_LE)
        }
    }

    @SuppressLint("MissingPermission")
    private fun disconnectFromDevice() {
        bluetoothGatt?.disconnect()
        bluetoothGatt?.close()
        bluetoothGatt = null
        isConnected.value = false
        esp32BatteryLevel.intValue = -1
        isEsp32Charging.value = false
        isSettingsOpen.value = false

        // 연결 정보 삭제
        prefs.edit {
            remove("lastDeviceAddress")
            remove("lastDeviceName")
        }
    }

    private val gattCallback = object : BluetoothGattCallback() {
        @SuppressLint("MissingPermission")
        override fun onConnectionStateChange(gatt: BluetoothGatt, status: Int, newState: Int) {
            if (newState == BluetoothProfile.STATE_CONNECTED) {
                gatt.discoverServices()
            } else if (newState == BluetoothProfile.STATE_DISCONNECTED) {
                isConnected.value = false
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
                        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                            gatt.writeDescriptor(descriptor, BluetoothGattDescriptor.ENABLE_NOTIFICATION_VALUE)
                        } else {
                            @Suppress("DEPRECATION")
                            descriptor.value = BluetoothGattDescriptor.ENABLE_NOTIFICATION_VALUE
                            @Suppress("DEPRECATION")
                            gatt.writeDescriptor(descriptor)
                        }
                    }
                }
            }
        }

        @SuppressLint("MissingPermission")
        override fun onDescriptorWrite(gatt: BluetoothGatt, descriptor: BluetoothGattDescriptor, status: Int) {
            if (status == BluetoothGatt.GATT_SUCCESS) {
                // Notification 설정이 완료된 후 연결 상태를 true로 변경
                isConnected.value = true
            }
        }

        @Suppress("DEPRECATION")
        @Deprecated("Deprecated in Java")
        override fun onCharacteristicRead(gatt: BluetoothGatt, characteristic: BluetoothGattCharacteristic, status: Int) {
            if (status == BluetoothGatt.GATT_SUCCESS) {
                val value = characteristic.value
                processReceivedData(value, isFromRead = true)
            }
        }

        override fun onCharacteristicChanged(gatt: BluetoothGatt, characteristic: BluetoothGattCharacteristic, value: ByteArray) {
            processReceivedData(value, isFromRead = false)
        }

        @Suppress("DEPRECATION")
        @Deprecated("Deprecated in Java")
        override fun onCharacteristicChanged(gatt: BluetoothGatt, characteristic: BluetoothGattCharacteristic) {
            val value = characteristic.value
            processReceivedData(value, isFromRead = false)
        }

        private fun processReceivedData(value: ByteArray, isFromRead: Boolean) {
            val receivedData = String(value)
            val parts = receivedData.split(",")
            if (parts.size < 2) return

            val type = parts[0].trim().uppercase()
            val dataValue = parts[1].trim()

            when (type) {
                "BATT" -> {
                    esp32BatteryLevel.intValue = dataValue.toIntOrNull() ?: -1
                    isEsp32Charging.value = if (parts.size >= 3) {
                        parts[2].trim().uppercase() == "CHARGING"
                    } else {
                        false
                    }
                }
                else -> {
                    if (!isFromRead) {
                        wakeUpScreen()
                        alertSoundType.value = type
                        alertAngle.floatValue = dataValue.toFloatOrNull() ?: 0f
                        initialAzimuth = currentAzimuth.floatValue

                        isSettingsOpen.value = false
                        isAlertActive.value = true
                        triggerHapticFeedback(this@MainActivity, vibeStrength.intValue)
                    }
                }
            }
        }
    }

    private fun wakeUpScreen() {
        if (wakeLock?.isHeld == false) {
            wakeLock?.acquire(3000)
        }
    }

    private fun triggerHapticFeedback(context: Context, level: Int) {
        val vibrator = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            val vibratorManager = context.getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as VibratorManager
            vibratorManager.defaultVibrator
        } else {
            @Suppress("DEPRECATION")
            context.getSystemService(Context.VIBRATOR_SERVICE) as Vibrator
        }

        val pattern = VIBRATE_PATTERNS.getOrElse(level) { VIBRATE_PATTERNS.last() }
        val effect = VibrationEffect.createWaveform(pattern, -1)
        vibrator.vibrate(effect)
    }

    @SuppressLint("MissingPermission")
    override fun onStop() {
        super.onStop()
        if (wakeLock?.isHeld == true) {
            wakeLock?.release()
        }
    }

    @SuppressLint("MissingPermission")
    override fun onDestroy() {
        super.onDestroy()
        scanner?.stopScan(scanCallback)
        bluetoothGatt?.close()
    }
}

// ---------------- UI 영역 ----------------

@SuppressLint("MissingPermission")
@Composable
fun ConnectScreen(
    isScanning: Boolean,
    devices: List<BluetoothDevice>,
    onStartScan: () -> Unit,
    onDeviceClick: (BluetoothDevice) -> Unit
) {
    Column(
        modifier = Modifier.fillMaxSize().background(Color.Black).padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        if (!isScanning && devices.isEmpty()) {
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
    esp32BatteryLevel: Int,
    isEsp32Charging: Boolean,
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
        DigitalClockScreen(
            clockColor = clockColor,
            esp32BatteryLevel = esp32BatteryLevel,
            isEsp32Charging = isEsp32Charging,
            onOpenSettings = onOpenSettings
        )
    }
}

@Composable
fun DigitalClockScreen(
    clockColor: Color,
    esp32BatteryLevel: Int,
    isEsp32Charging: Boolean,
    onOpenSettings: () -> Unit
) {
    var currentTime by remember { mutableStateOf("") }
    var batteryLevel by remember { mutableIntStateOf(100) }
    var isCharging by remember { mutableStateOf(false) }
    val context = androidx.compose.ui.platform.LocalContext.current

    // 시계 업데이트 (1초마다)
    LaunchedEffect(Unit) {
        while (true) {
            val sdf = SimpleDateFormat("HH:mm:ss", Locale.getDefault())
            currentTime = sdf.format(Date())
            delay(1000.milliseconds)
        }
    }

    // 배터리 상태 업데이트 (이벤트 발생 시에만)
    DisposableEffect(Unit) {
        val receiver = object : android.content.BroadcastReceiver() {
            override fun onReceive(context: Context?, intent: Intent?) {
                intent?.let {
                    val level = it.getIntExtra(BatteryManager.EXTRA_LEVEL, -1)
                    val scale = it.getIntExtra(BatteryManager.EXTRA_SCALE, -1)
                    if (level != -1 && scale != -1) {
                        batteryLevel = (level * 100 / scale.toFloat()).toInt()
                    }

                    val status = it.getIntExtra(BatteryManager.EXTRA_STATUS, -1)
                    isCharging = status == BatteryManager.BATTERY_STATUS_CHARGING ||
                            status == BatteryManager.BATTERY_STATUS_FULL
                }
            }
        }
        val filter = IntentFilter(Intent.ACTION_BATTERY_CHANGED)
        context.registerReceiver(receiver, filter)

        onDispose {
            context.unregisterReceiver(receiver)
        }
    }

    Box(
        modifier = Modifier.fillMaxSize().background(Color.Black),
        contentAlignment = Alignment.Center
    ) {
        // 설정 버튼
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

        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            modifier = Modifier.offset(y = (-5).dp)
        ) {
            val batteryIcon = when {
                isCharging -> Icons.Default.BatteryChargingFull
                batteryLevel <= 15 -> Icons.Default.BatteryAlert
                batteryLevel <= 25 -> Icons.Default.Battery0Bar
                batteryLevel <= 35 -> Icons.Default.Battery1Bar
                batteryLevel <= 45 -> Icons.Default.Battery2Bar
                batteryLevel <= 55 -> Icons.Default.Battery3Bar
                batteryLevel <= 70 -> Icons.Default.Battery4Bar
                batteryLevel <= 85 -> Icons.Default.Battery5Bar
                batteryLevel <= 95 -> Icons.Default.Battery6Bar
                else -> Icons.Default.BatteryFull
            }

            Row(verticalAlignment = Alignment.CenterVertically) {
                // 워치 배터리
                Icon(
                    imageVector = batteryIcon,
                    contentDescription = "Watch Battery",
                    tint = Color.White,
                    modifier = Modifier.size(14.dp)
                )
                Spacer(modifier = Modifier.width(4.dp))
                Text(
                    text = "$batteryLevel%",
                    color = Color.White,
                    fontSize = 12.sp,
                    fontWeight = FontWeight.Medium
                )

                // ESP32 배터리가 있는 경우에만 표시
                if (esp32BatteryLevel != -1) {
                    Spacer(modifier = Modifier.width(12.dp))

                    val esp32BatteryIcon = when {
                        isEsp32Charging -> Icons.Default.BatteryChargingFull
                        esp32BatteryLevel <= 15 -> Icons.Default.BatteryAlert
                        esp32BatteryLevel <= 25 -> Icons.Default.Battery0Bar
                        esp32BatteryLevel <= 35 -> Icons.Default.Battery1Bar
                        esp32BatteryLevel <= 45 -> Icons.Default.Battery2Bar
                        esp32BatteryLevel <= 55 -> Icons.Default.Battery3Bar
                        esp32BatteryLevel <= 70 -> Icons.Default.Battery4Bar
                        esp32BatteryLevel <= 85 -> Icons.Default.Battery5Bar
                        esp32BatteryLevel <= 95 -> Icons.Default.Battery6Bar
                        else -> Icons.Default.BatteryFull
                    }

                    Icon(
                        imageVector = esp32BatteryIcon,
                        contentDescription = "ESP32 Battery",
                        tint = Color(0xFF00E5FF), // ESP32 배터리는 다른 색상으로 구분
                        modifier = Modifier.size(14.dp)
                    )
                    Spacer(modifier = Modifier.width(4.dp))
                    Text(
                        text = "$esp32BatteryLevel%",
                        color = Color(0xFF00E5FF),
                        fontSize = 12.sp,
                        fontWeight = FontWeight.Medium
                    )
                }
            }

            Spacer(modifier = Modifier.height(6.dp))

            Text(
                text = currentTime,
                color = clockColor,
                fontSize = 36.sp,
                fontWeight = FontWeight.Bold
            )
        }
    }
}

@Composable
fun AlertScreen(soundType: String, angle: Float, themeColor: Color) {
    val cleanSoundType = soundType.trim().uppercase()
    val alertIcon = if (cleanSoundType == "SIREN") Icons.Default.NotificationsActive else Icons.AutoMirrored.Filled.VolumeUp

    Box(modifier = Modifier.fillMaxSize().background(Color.Black), contentAlignment = Alignment.Center) {
        Canvas(modifier = Modifier.fillMaxSize()) {
            val strokeWidth = 25f
            drawCircle(color = Color.DarkGray, style = Stroke(width = strokeWidth))
            drawArc(
                color = themeColor,
                startAngle = angle - 90f - 30f,
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
