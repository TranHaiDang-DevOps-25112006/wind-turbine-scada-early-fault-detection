# Data Dictionary (Xac minh thuc te tu Wind Farm A - CARE To Compare)

## 1. Thong tin tep mau khao sat
- **Tep SCADA khao sat:** `Wind Farm A\Wind Farm A\datasets\0.csv`
- **Dau phan cach cot (Delimiter):** `;`
- **So dong cua tep mau:** 54,986 dong
- **Tong so cot:** 86 cot

## 2. Bang tu dien cac truong trong tep SCADA

| Ten cot thuc te | Y nghia | Kieu du lieu | Don vi | Thieu (%) | Duoc dung lam feature? | Ghi chu |
|---|---|---|---|---:|---|---|
| `time_stamp` | Thoi diem ghi nhan ban ghi SCADA (tan suat 10 phut) | `object` | YYYY-MM-DD HH:MM:SS | 0.00% | Khong (Cot thoi gian) | Metadata chuan cua dataset |
| `asset_id` | Ma dinh danh tua-bin gio (Turbine ID) | `int64` | ID | 0.00% | Khong (Cot thiet bi) | Metadata chuan cua dataset |
| `id` | Ma thu tu dong ban ghi trong dataset | `int64` | ID | 0.00% | Khong (Cot dinh danh) | Metadata chuan cua dataset |
| `train_test` | Nhan phan chia tap du lieu (train / prediction) | `object` | Category | 0.00% | Khong (Dung chia tap du lieu) | Metadata chuan cua dataset |
| `status_type_id` | Ma trang thai van hanh tua-bin (0: binh thuong) | `int64` | Code (0-5) | 0.00% | Khong (Trang thai van hanh) | Metadata chuan cua dataset |
| `sensor_0_avg` | Ambient temperature | `float64` | �C | 0.00% | Co | Bien cam bien dau vao cho mo hinh |
| `sensor_1_avg` | Wind absolute direction | `float64` | � | 0.00% | Co | Bien cam bien dau vao cho mo hinh |
| `sensor_2_avg` | Wind relative direction | `float64` | � | 0.00% | Co | Bien cam bien dau vao cho mo hinh |
| `wind_speed_3_avg` | Windspeed | `float64` | m/s | 0.00% | Co | Bien cam bien dau vao cho mo hinh |
| `wind_speed_4_avg` | Estimated windspeed | `float64` | m/s | 0.00% | Co | Bien cam bien dau vao cho mo hinh |
| `wind_speed_3_max` | Windspeed | `float64` | m/s | 0.00% | Co | Bien cam bien dau vao cho mo hinh |
| `wind_speed_3_min` | Windspeed | `float64` | m/s | 0.00% | Co | Bien cam bien dau vao cho mo hinh |
| `wind_speed_3_std` | Windspeed | `float64` | m/s | 0.00% | Co | Bien cam bien dau vao cho mo hinh |
| `sensor_5_avg` | Pitch angle | `float64` | � | 0.00% | Co | Bien cam bien dau vao cho mo hinh |
| `sensor_5_max` | Pitch angle | `float64` | � | 0.00% | Co | Bien cam bien dau vao cho mo hinh |
| `sensor_5_min` | Pitch angle | `float64` | � | 0.00% | Co | Bien cam bien dau vao cho mo hinh |
| `sensor_5_std` | Pitch angle | `float64` | � | 0.00% | Co | Bien cam bien dau vao cho mo hinh |
| `sensor_6_avg` | Temperature in the hub controller | `float64` | �C | 0.00% | Co | Bien cam bien dau vao cho mo hinh |
| `sensor_7_avg` | Temperature in the top nacelle controller | `float64` | �C | 0.00% | Co | Bien cam bien dau vao cho mo hinh |
| `sensor_8_avg` | Temperature in the choke coils on the VCS-section | `float64` | �C | 0.00% | Co | Bien cam bien dau vao cho mo hinh |
| `sensor_9_avg` | Temperature on the VCP-board | `float64` | �C | 0.00% | Co | Bien cam bien dau vao cho mo hinh |
| `sensor_10_avg` | Temperature in the VCS cooling water | `float64` | �C | 0.00% | Co | Bien cam bien dau vao cho mo hinh |
| `sensor_11_avg` | Temperature in gearbox bearing on high speed shaft | `float64` | �C | 0.00% | Co | Bien cam bien dau vao cho mo hinh |
| `sensor_12_avg` | Temperature oil in gearbox | `float64` | �C | 0.00% | Co | Bien cam bien dau vao cho mo hinh |
| `sensor_13_avg` | Temperature in generator bearing 2 (Drive End) | `float64` | �C | 0.00% | Co | Bien cam bien dau vao cho mo hinh |
| `sensor_14_avg` | Temperature in generator bearing 1 (Non-Drive End) | `float64` | �C | 0.00% | Co | Bien cam bien dau vao cho mo hinh |
| `sensor_15_avg` | Temperature inside generator in stator windings phase 1 | `float64` | �C | 0.00% | Co | Bien cam bien dau vao cho mo hinh |
| `sensor_16_avg` | Temperature inside generator in stator windings phase 2 | `float64` | �C | 0.00% | Co | Bien cam bien dau vao cho mo hinh |
| `sensor_17_avg` | Temperature inside generator in stator windings phase 3 | `float64` | �C | 0.00% | Co | Bien cam bien dau vao cho mo hinh |
| `sensor_18_avg` | Generator rpm in latest period | `float64` | rpm | 0.00% | Co | Bien cam bien dau vao cho mo hinh |
| `sensor_18_max` | Generator rpm in latest period | `float64` | rpm | 0.00% | Co | Bien cam bien dau vao cho mo hinh |
| `sensor_18_min` | Generator rpm in latest period | `float64` | rpm | 0.00% | Co | Bien cam bien dau vao cho mo hinh |
| `sensor_18_std` | Generator rpm in latest period | `float64` | rpm | 0.00% | Co | Bien cam bien dau vao cho mo hinh |
| `sensor_19_avg` | Temperature in the split ring chamber | `float64` | �C | 0.00% | Co | Bien cam bien dau vao cho mo hinh |
| `sensor_20_avg` | Temperature in the busbar section | `float64` | �C | 0.00% | Co | Bien cam bien dau vao cho mo hinh |
| `sensor_21_avg` | Temperature measured by the IGBT-driver on the grid side inverter | `float64` | �C | 0.00% | Co | Bien cam bien dau vao cho mo hinh |
| `sensor_22_avg` | Actual phase displacement | `float64` | nan | 0.00% | Co | Bien cam bien dau vao cho mo hinh |
| `sensor_23_avg` | Averaged current in phase 1 | `float64` | A | 0.00% | Co | Bien cam bien dau vao cho mo hinh |
| `sensor_24_avg` | Averaged current in phase 2 | `float64` | A | 0.00% | Co | Bien cam bien dau vao cho mo hinh |
| `sensor_25_avg` | Averaged current in phase 3 | `float64` | A | 0.00% | Co | Bien cam bien dau vao cho mo hinh |
| `sensor_26_avg` | Grid frequency | `float64` | Hz | 0.00% | Co | Bien cam bien dau vao cho mo hinh |
| `reactive_power_27_avg` | Possible grid capacitive reactive power | `float64` | kVAr | 0.00% | Co | Bien cam bien dau vao cho mo hinh |
| `reactive_power_27_max` | Possible grid capacitive reactive power | `float64` | kVAr | 0.00% | Co | Bien cam bien dau vao cho mo hinh |
| `reactive_power_27_min` | Possible grid capacitive reactive power | `float64` | kVAr | 0.00% | Co | Bien cam bien dau vao cho mo hinh |
| `reactive_power_27_std` | Possible grid capacitive reactive power | `float64` | kVAr | 0.00% | Co | Bien cam bien dau vao cho mo hinh |
| `reactive_power_28_avg` | Possible grid inductive reactive power | `float64` | kVAr | 0.00% | Co | Bien cam bien dau vao cho mo hinh |
| `reactive_power_28_max` | Possible grid inductive reactive power | `float64` | kVAr | 0.00% | Co | Bien cam bien dau vao cho mo hinh |
| `reactive_power_28_min` | Possible grid inductive reactive power | `float64` | kVAr | 0.00% | Co | Bien cam bien dau vao cho mo hinh |
| `reactive_power_28_std` | Possible grid inductive reactive power | `float64` | kVAr | 0.00% | Co | Bien cam bien dau vao cho mo hinh |
| `power_29_avg` | Possible grid active power | `float64` | kW | 0.00% | Co | Bien cam bien dau vao cho mo hinh |
| `power_29_max` | Possible grid active power | `float64` | kW | 0.00% | Co | Bien cam bien dau vao cho mo hinh |
| `power_29_min` | Possible grid active power | `float64` | kW | 0.00% | Co | Bien cam bien dau vao cho mo hinh |
| `power_29_std` | Possible grid active power | `float64` | kW | 0.00% | Co | Bien cam bien dau vao cho mo hinh |
| `power_30_avg` | Grid power | `float64` | kW | 0.00% | Co | Bien cam bien dau vao cho mo hinh |
| `power_30_max` | Grid power | `float64` | kW | 0.00% | Co | Bien cam bien dau vao cho mo hinh |
| `power_30_min` | Grid power | `float64` | kW | 0.00% | Co | Bien cam bien dau vao cho mo hinh |
| `power_30_std` | Grid power | `float64` | kW | 0.00% | Co | Bien cam bien dau vao cho mo hinh |
| `sensor_31_avg` | Grid reactive power | `float64` | kVAr | 0.00% | Co | Bien cam bien dau vao cho mo hinh |
| `sensor_31_max` | Grid reactive power | `float64` | kVAr | 0.00% | Co | Bien cam bien dau vao cho mo hinh |
| `sensor_31_min` | Grid reactive power | `float64` | kVAr | 0.00% | Co | Bien cam bien dau vao cho mo hinh |
| `sensor_31_std` | Grid reactive power | `float64` | kVAr | 0.00% | Co | Bien cam bien dau vao cho mo hinh |
| `sensor_32_avg` | Averaged voltage in phase 1 | `float64` | V | 0.00% | Co | Bien cam bien dau vao cho mo hinh |
| `sensor_33_avg` | Averaged voltage in phase 2 | `float64` | V | 0.00% | Co | Bien cam bien dau vao cho mo hinh |
| `sensor_34_avg` | Averaged voltage in phase 3 | `float64` | V | 0.00% | Co | Bien cam bien dau vao cho mo hinh |
| `sensor_35_avg` | Temperature measured by the IGBT-driver on the rotor side inverter phase1 | `float64` | �C | 0.00% | Co | Bien cam bien dau vao cho mo hinh |
| `sensor_36_avg` | Temperature measured by the IGBT-driver on the rotor side inverter phase2 | `float64` | �C | 0.00% | Co | Bien cam bien dau vao cho mo hinh |
| `sensor_37_avg` | Temperature measured by the IGBT-driver on the rotor side inverter phase3 | `float64` | �C | 0.00% | Co | Bien cam bien dau vao cho mo hinh |
| `sensor_38_avg` | Temperature in HV transformer phase L1 | `float64` | �C | 0.00% | Co | Bien cam bien dau vao cho mo hinh |
| `sensor_39_avg` | Temperature in HV transformer phase L2 | `float64` | �C | 0.00% | Co | Bien cam bien dau vao cho mo hinh |
| `sensor_40_avg` | Temperature in HV transformer phase L3 | `float64` | �C | 0.00% | Co | Bien cam bien dau vao cho mo hinh |
| `sensor_41_avg` | Temperature oil in hydraulic group | `float64` | �C | 0.00% | Co | Bien cam bien dau vao cho mo hinh |
| `sensor_42_avg` | Nacelle direction | `float64` | � | 0.00% | Co | Bien cam bien dau vao cho mo hinh |
| `sensor_43_avg` | Nacelle temperature | `float64` | �C | 0.00% | Co | Bien cam bien dau vao cho mo hinh |
| `sensor_44` | Active power - generator disconnected | `float64` | Wh | 0.00% | Co | Bien cam bien dau vao cho mo hinh |
| `sensor_45` | Active power - generator connected in delta | `float64` | Wh | 0.00% | Co | Bien cam bien dau vao cho mo hinh |
| `sensor_46` | Active power - generator connected in star | `float64` | Wh | 0.00% | Co | Bien cam bien dau vao cho mo hinh |
| `sensor_47` | Reactive power - generator disconnected | `float64` | VArh | 0.00% | Co | Bien cam bien dau vao cho mo hinh |
| `sensor_48` | Reactive power - generator connected in delta | `float64` | VArh | 0.00% | Co | Bien cam bien dau vao cho mo hinh |
| `sensor_49` | Reactive power - generator connected in star | `float64` | VArh | 0.00% | Co | Bien cam bien dau vao cho mo hinh |
| `sensor_50` | Total active power | `float64` | Wh | 0.00% | Co | Bien cam bien dau vao cho mo hinh |
| `sensor_51` | Total reactive power | `float64` | VArh | 0.00% | Co | Bien cam bien dau vao cho mo hinh |
| `sensor_52_avg` | Rotor rpm | `float64` | rpm | 0.00% | Co | Bien cam bien dau vao cho mo hinh |
| `sensor_52_max` | Rotor rpm | `float64` | rpm | 0.00% | Co | Bien cam bien dau vao cho mo hinh |
| `sensor_52_min` | Rotor rpm | `float64` | rpm | 0.00% | Co | Bien cam bien dau vao cho mo hinh |
| `sensor_52_std` | Rotor rpm | `float64` | rpm | 0.00% | Co | Bien cam bien dau vao cho mo hinh |
| `sensor_53_avg` | Temperature in the nose cone | `float64` | �C | 0.00% | Co | Bien cam bien dau vao cho mo hinh |

## 3. Cau truc tep nhan su kien (event_info.csv)
- **Duong dan:** `Wind Farm A\Wind Farm A\event_info.csv` (22 su kien)
- **Cac cot thuc te:** `event_id, event_label, event_start, event_start_id, event_end, event_end_id, event_description`
- **Dac diem nhan:** Nhan khong gan truc tiep tung dong ma dinh nghia theo khoang thoi gian su kien (event_start den event_end) kem event_description.