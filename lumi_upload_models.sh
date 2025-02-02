#!/bin/bash -e


python -m vec2text.upload_model mbert_multihplt_500 yiyic/mbert_multihplt_500

timeout /t 2 /nobreak >nul


python -m vec2text.upload_model gtr_multihplt_300 yiyic/gtr_multihplt_300

timeout /t 2 /nobreak >nul

python -m vec2text.upload_model t5_multihplt_300 yiyic/t5_multihplt_300

timeout /t 2 /nobreak >nul


python -m vec2text.upload_model mt5_multihplt_300 yiyic/mt5_multihplt_300

timeout /t 2 /nobreak >nul


python -m vec2text.upload_model mbert_multihplt_300 yiyic/mbert_multihplt_300

timeout /t 2 /nobreak >nul


python -m vec2text.upload_model gtr_multihplt_100 yiyic/gtr_multihplt_100

timeout /t 2 /nobreak >nul



python -m vec2text.upload_model t5_multihplt_100 yiyic/t5_multihplt_100

timeout /t 2 /nobreak >nul


python -m vec2text.upload_model mt5_multihplt_100 yiyic/mt5_multihplt_100

timeout /t 2 /nobreak >nul


python -m vec2text.upload_model mbert_multihplt_100 yiyic/mbert_multihplt_100

timeout /t 2 /nobreak >nul



python -m vec2text.upload_model gtr_multihplt_30 yiyic/gtr_multihplt_30

timeout /t 2 /nobreak >nul



python -m vec2text.upload_model t5_multihplt_30 yiyic/t5_multihplt_30

timeout /t 2 /nobreak >nul



python -m vec2text.upload_model mt5_multihplt_30 yiyic/mt5_multihplt_30

timeout /t 2 /nobreak >nul


python -m vec2text.upload_model gtr_multihplt_10 yiyic/gtr_multihplt_10

timeout /t 2 /nobreak >nul


python -m vec2text.upload_model t5_multihplt_10 yiyic/t5_multihplt_10

timeout /t 2 /nobreak >nul


python -m vec2text.upload_model mt5_multihplt_10 yiyic/mt5_multihplt_10

timeout /t 2 /nobreak >nul



python -m vec2text.upload_model mbert_multihplt_10 yiyic/mbert_multihplt_10

timeout /t 2 /nobreak >nul


