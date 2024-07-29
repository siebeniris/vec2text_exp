#!/bin/bash -e


python -m vec2text.upload_model urd_pan_32_2layers_corrector mt5_me5_urd_pan_32_2layers_corrector


sleep 5

python -m vec2text.upload_model urd_guj_32_2layers_corrector mt5_me5_urd_guj_32_2layers_corrector



sleep 5

python -m vec2text.upload_model urd_hin_32_2layers_corrector mt5_me5_urd_hin_32_2layers_corrector



sleep 5

python -m vec2text.upload_model hin_pan_32_2layers_corrector mt5_me5_hin_pan_32_2layers_corrector



sleep 5

python -m vec2text.upload_model pan_guj_32_2layers_corrector mt5_me5_pan_guj_32_2layers_corrector



sleep 5

python -m vec2text.upload_model tur_urd_32_2layers_corrector mt5_me5_tur_urd_32_2layers_corrector




sleep 5

python -m vec2text.upload_model tur_pan_32_2layers_corrector mt5_me5_tur_pan_32_2layers_corrector



sleep 5

python -m vec2text.upload_model tur_guj_32_2layers_corrector mt5_me5_tur_guj_32_2layers_corrector


sleep 5

python -m vec2text.upload_model kaz_pan_32_2layers_corrector mt5_me5_kaz_pan_32_2layers_corrector


sleep 5

python -m vec2text.upload_model kaz_guj_32_2layers_corrector mt5_me5_kaz_guj_32_2layers_corrector



sleep 5

python -m vec2text.upload_model kaz_hin_32_2layers_corrector mt5_me5_kaz_hin_32_2layers_corrector