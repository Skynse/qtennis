#include <Wire.h>

// ICM-20948 on the Adafruit QT Py ESP32-S3 STEMMA QT port.
// On the QT Py S3 that port is the second I2C bus (Wire1): SDA1 = GPIO41, SCL1 = GPIO40.
#define STEMMA_SDA 41
#define STEMMA_SCL 40
TwoWire *bus = &Wire1;

// Adafruit ICM-20948 defaults to 0x69; 0x68 if the address jumper is changed
const uint8_t ICM_ADDR_CANDIDATES[] = {0x69, 0x68};
uint8_t icmAddr = 0;

bool findIcm() {
  for (uint8_t addr : ICM_ADDR_CANDIDATES) {
    bus->beginTransmission(addr);
    if (bus->endTransmission() == 0) { icmAddr = addr; return true; }
  }
  return false;
}

void scanBus() {
  Serial.print("I2C scan on STEMMA QT:");
  int n = 0;
  for (uint8_t a = 1; a < 127; a++) {
    bus->beginTransmission(a);
    if (bus->endTransmission() == 0) { Serial.printf(" 0x%02X", a); n++; }
  }
  Serial.println(n ? "" : " nothing");
}

// Bank 0 registers
#define REG_WHO_AM_I     0x00
#define REG_PWR_MGMT_1   0x06
#define REG_PWR_MGMT_2   0x07
#define REG_ACCEL_XOUT_H 0x2D   // accel(6) gyro(6) temp(2) are contiguous
#define REG_BANK_SEL     0x7F
#define ICM_WHO_AM_I_VAL 0xEA

#define REG_USER_CTRL    0x03   // [5] I2C_MST_EN
#define REG_INT_PIN_CFG  0x0F   // [1] BYPASS_EN

// AK09916 magnetometer (inside the ICM-20948 package, on its aux I2C bus).
// With BYPASS_EN set, the aux bus is bridged onto the main bus and the AK09916 answers at 0x0C.
#define AK_ADDR        0x0C
#define AK_WIA2        0x01   // device ID, reads 0x09
#define AK_ST1         0x10   // [0] DRDY
#define AK_HXL         0x11   // HXL..HZH (little endian), TMPS, ST2 follow
#define AK_CNTL2       0x31   // mode
#define AK_CNTL3       0x32   // [0] SRST
#define AK_WIA2_VAL    0x09
#define AK_MODE_100HZ  0x08
const float MAG_UT_PER_LSB = 0.15f;
bool magOk = false;
float mx = 0, my = 0, mz = 0;   // last magnetometer reading, uT, in the accel/gyro frame

// Bank 2 registers
#define REG_GYRO_CONFIG_1  0x01   // [2:1] FS_SEL, [0] DLPF enable
#define REG_ACCEL_CONFIG   0x14   // [2:1] FS_SEL, [0] DLPF enable

// Full-scale ranges set in setup(): accel +-16 g, gyro +-2000 dps (swings saturate anything less)
const float ACCEL_LSB_PER_G   = 2048.0f;
const float GYRO_LSB_PER_DPS  = 16.4f;

const uint32_t SAMPLE_PERIOD_US = 5000;   // 200 Hz

bool writeReg(uint8_t reg, uint8_t val) {
  bus->beginTransmission(icmAddr);
  bus->write(reg);
  bus->write(val);
  return bus->endTransmission() == 0;
}

bool readRegs(uint8_t reg, uint8_t *buf, size_t len) {
  bus->beginTransmission(icmAddr);
  bus->write(reg);
  if (bus->endTransmission(false) != 0) return false;
  if (bus->requestFrom(icmAddr, (uint8_t)len) != len) return false;
  for (size_t i = 0; i < len; i++) buf[i] = bus->read();
  return true;
}

void selectBank(uint8_t bank) {
  writeReg(REG_BANK_SEL, bank << 4);
}

bool akWrite(uint8_t reg, uint8_t val) {
  bus->beginTransmission(AK_ADDR);
  bus->write(reg);
  bus->write(val);
  return bus->endTransmission() == 0;
}

bool akRead(uint8_t reg, uint8_t *buf, size_t len) {
  bus->beginTransmission(AK_ADDR);
  bus->write(reg);
  if (bus->endTransmission(false) != 0) return false;
  if (bus->requestFrom((uint8_t)AK_ADDR, (uint8_t)len) != len) return false;
  for (size_t i = 0; i < len; i++) buf[i] = bus->read();
  return true;
}

bool setupMag() {
  selectBank(0);
  writeReg(REG_USER_CTRL, 0x00);     // ICM's own I2C master off, so it doesn't drive the aux bus
  writeReg(REG_INT_PIN_CFG, 0x02);   // BYPASS_EN: AK09916 appears on our bus
  delay(10);

  uint8_t wia = 0;
  if (!akRead(AK_WIA2, &wia, 1) || wia != AK_WIA2_VAL) {
    Serial.printf("AK09916 not found (WIA2=0x%02X) - continuing without magnetometer\n", wia);
    return false;
  }
  akWrite(AK_CNTL3, 0x01);           // soft reset
  delay(100);
  akWrite(AK_CNTL2, AK_MODE_100HZ);  // continuous measurement, 100 Hz
  delay(10);
  Serial.println("AK09916 OK (100 Hz)");
  return true;
}

// Poll the magnetometer; updates mx/my/mz when a new sample is ready.
void readMag() {
  uint8_t st1 = 0;
  if (!akRead(AK_ST1, &st1, 1) || !(st1 & 0x01)) return;
  uint8_t b[8];                      // HXL..HZH, TMPS, ST2 (reading ST2 releases the data registers)
  if (!akRead(AK_HXL, b, sizeof(b))) return;
  if (b[7] & 0x08) return;           // HOFL: magnetic overflow, discard
  int16_t x = (int16_t)(b[1] << 8 | b[0]);
  int16_t y = (int16_t)(b[3] << 8 | b[2]);
  int16_t z = (int16_t)(b[5] << 8 | b[4]);
  // AK09916 axes vs accel/gyro axes (ICM-20948 datasheet): X same, Y and Z inverted
  mx =  x * MAG_UT_PER_LSB;
  my = -y * MAG_UT_PER_LSB;
  mz = -z * MAG_UT_PER_LSB;
}

void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10);

  bus->begin(STEMMA_SDA, STEMMA_SCL, 400000);   // ICM-20948 I2C max is 400 kHz

  if (!findIcm()) {
    Serial.println("ICM-20948 not found at 0x68/0x69 on STEMMA QT (SDA=41, SCL=40)");
    scanBus();
    Serial.println("Check: STEMMA QT cable seated at both ends, address jumper untouched");
    while (true) delay(1000);
  }
  Serial.printf("ICM-20948 ACK at 0x%02X\n", icmAddr);

  // Soft reset (DEVICE_RESET bit), then wake with auto clock select
  selectBank(0);
  writeReg(REG_PWR_MGMT_1, 0x80);
  delay(100);
  selectBank(0);
  writeReg(REG_PWR_MGMT_1, 0x01);
  delay(30);
  writeReg(REG_PWR_MGMT_2, 0x00);   // enable all accel + gyro axes
  Serial.println("Sent soft reset");

  selectBank(2);
  writeReg(REG_GYRO_CONFIG_1, (3 << 1) | 1);   // +-2000 dps, DLPF on
  writeReg(REG_ACCEL_CONFIG,  (3 << 1) | 1);   // +-16 g, DLPF on
  selectBank(0);

  uint8_t who = 0;
  readRegs(REG_WHO_AM_I, &who, 1);
  Serial.printf("WHO_AM_I = 0x%02X (%s)\n", who,
                who == ICM_WHO_AM_I_VAL ? "ICM-20948 OK" : "unexpected");
  if (who != ICM_WHO_AM_I_VAL) while (true) delay(1000);

  magOk = setupMag();
}

void loop() {
  uint8_t raw[14];
  if (!readRegs(REG_ACCEL_XOUT_H, raw, sizeof(raw))) {
    Serial.println("I2C read failed");
    delay(100);
    return;
  }

  int16_t v[7];
  for (int i = 0; i < 7; i++) v[i] = (int16_t)((raw[2 * i] << 8) | raw[2 * i + 1]);

  float ax = v[0] / ACCEL_LSB_PER_G,  ay = v[1] / ACCEL_LSB_PER_G,  az = v[2] / ACCEL_LSB_PER_G;
  float gx = v[3] / GYRO_LSB_PER_DPS, gy = v[4] / GYRO_LSB_PER_DPS, gz = v[5] / GYRO_LSB_PER_DPS;
  float tempC = (v[6] - 21.0f) / 333.87f + 21.0f;

  (void)tempC;

  // Machine-readable line for the host visualizer:
  //   D,<micros>,ax,ay,az[g],gx,gy,gz[dps]            (no magnetometer)
  //   D,<micros>,ax,ay,az[g],gx,gy,gz[dps],mx,my,mz[uT]
  if (magOk) {
    readMag();
    Serial.printf("D,%lu,%.4f,%.4f,%.4f,%.3f,%.3f,%.3f,%.2f,%.2f,%.2f\n",
                  (unsigned long)micros(), ax, ay, az, gx, gy, gz, mx, my, mz);
  } else {
    Serial.printf("D,%lu,%.4f,%.4f,%.4f,%.3f,%.3f,%.3f\n",
                  (unsigned long)micros(), ax, ay, az, gx, gy, gz);
  }

  static uint32_t next = micros();
  next += SAMPLE_PERIOD_US;
  int32_t wait = (int32_t)(next - micros());
  if (wait > 0) delayMicroseconds(wait);
  else next = micros();   // fell behind; resync
}
