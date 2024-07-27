#!/bin/bash -e


python -m vec2text.upload_model tur_urd_32_2layers_inverter mt5_me5_tur_urd_32_2layers_inverter


sleep 5

python -m vec2text.upload_model tur_pan_32_2layers_inverter mt5_me5_tur_pan_32_2layers_inverter



sleep 5

python -m vec2text.upload_model tur_guj_32_2layers_inverter mt5_me5_tur_guj_32_2layers_inverter



sleep 5

python -m vec2text.upload_model tur_hin_32_2layers_inverter mt5_me5_tur_hin_32_2layers_inverter



sleep 5

python -m vec2text.upload_model kaz_urd_32_2layers_inverter mt5_me5_kaz_urd_32_2layers_inverter



sleep 5

python -m vec2text.upload_model kaz_pan_32_2layers_inverter mt5_me5_kaz_pan_32_2layers_inverter




sleep 5

python -m vec2text.upload_model kaz_guj_32_2layers_inverter mt5_me5_kaz_guj_32_2layers_inverter



sleep 5

python -m vec2text.upload_model kaz_hin_32_2layers_inverter mt5_me5_kaz_hin_32_2layers_inverter


