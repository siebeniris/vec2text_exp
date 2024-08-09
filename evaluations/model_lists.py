model_list_inverter = [
    # deu_Latn, hebr_Hebr, cmn_Hani
    'mt5_me5_deu_Latn_32_2layers_inverter', 'mt5_me5_heb_Hebr_32_2layers_inverter',
    'mt5_me5_cmn_Hani_32_2layers_inverter',
    'mt5_me5_arb_Arab_32_2layers_inverter', 'mt5_me5_jpn_Jpan_32_2layers_inverter',
    'mt5_me5_tur_Latn_32_2layers_inverter', 'mt5_me5_kaz_Cyrl_32_2layers_inverter',
    'mt5_me5_mon_Cyrl_32_2layers_inverter', 'mt5_me5_urd_Arab_32_2layers_inverter',
    'mt5_me5_pan_Guru_32_2layers_inverter', 'mt5_me5_guj_Gujr_32_2layers_inverter',
    'mt5_me5_hin_Deva_32_2layers_inverter',

    # script
    'mt5_me5_turkic-fami_32_2layers_inverter',
    'mt5_me5_arab-script_32_2layers_inverter',
    'mt5_me5_latn-script_32_2layers_inverter',
    'mt5_me5_cyrl-script_32_2layers_inverter',
    'mt5_me5_cmn_jpn_32_2layers_inverter',

    # family
    'mt5_me5_heb_arb_32_2layers_inverter', 'mt5_me5_urd_pan_32_2layers_inverter',
    'mt5_me5_urd_guj_32_2layers_inverter', 'mt5_me5_urd_hin_32_2layers_inverter',
    'mt5_me5_hin_pan_32_2layers_inverter', 'mt5_me5_hin_guj_32_2layers_inverter',
    'mt5_me5_pan_guj_32_2layers_inverter',

    # random.
    'mt5_me5_tur_urd_32_2layers_inverter',
    'mt5_me5_tur_pan_32_2layers_inverter', 'mt5_me5_tur_guj_32_2layers_inverter',
    'mt5_me5_tur_hin_32_2layers_inverter', 'mt5_me5_kaz_urd_32_2layers_inverter',
    'mt5_me5_kaz_pan_32_2layers_inverter', 'mt5_me5_kaz_guj_32_2layers_inverter',
    'mt5_me5_kaz_hin_32_2layers_inverter'
]

model_list_corrector = [
    # monolingual
    'mt5_me5_deu_Latn_32_2layers_corrector', 'mt5_me5_heb_Hebr_32_2layers_corrector',
    'mt5_me5_cmn_Hani_32_2layers_corrector',
    'mt5_me5_arb_Arab_32_2layers_corrector', 'mt5_me5_jpn_Jpan_32_2layers_corrector',
    'mt5_me5_tur_Latn_32_2layers_corrector', 'mt5_me5_kaz_Cyrl_32_2layers_corrector',
    'mt5_me5_mon_Cyrl_32_2layers_corrector', 'mt5_me5_urd_Arab_32_2layers_corrector',
    'mt5_me5_pan_Guru_32_2layers_corrector', 'mt5_me5_guj_Gujr_32_2layers_corrector',
    'mt5_me5_hin_Deva_32_2layers_corrector',

    # script
    'mt5_me5_ara-script_32_2layers_corrector',
    'mt5_me5_latn-script_32_2layers_corrector',
    'mt5_me5_cyrl-script_32_2layers_corrector',
    'mt5_me5_cmn_jpn_32_2layers_corrector',

    # family
    'mt5_me5_turkic-fami_32_2layers_corrector',
    'mt5_me5_heb_arb_32_2layers_corrector', 'mt5_me5_urd_pan_32_2layers_corrector',
    'mt5_me5_urd_guj_32_2layers_corrector', 'mt5_me5_urd_hin_32_2layers_corrector',
    'mt5_me5_hin_pan_32_2layers_corrector', 'mt5_me5_hin_guj_32_2layers_corrector',
    'mt5_me5_pan_guj_32_2layers_corrector',

    # random.
    'mt5_me5_tur_urd_32_2layers_corrector',
    'mt5_me5_tur_pan_32_2layers_corrector', 'mt5_me5_tur_guj_32_2layers_corrector',
    'mt5_me5_tur_hin_32_2layers_corrector', 'mt5_me5_kaz_urd_32_2layers_corrector',
    'mt5_me5_kaz_pan_32_2layers_corrector', 'mt5_me5_kaz_guj_32_2layers_corrector',
    'mt5_me5_kaz_hin_32_2layers_corrector']

model_list_inverter_mono = ["mt5_gtr_deu_Latn_32_inverter", "mt5_alephbert_heb_Hebr_32_inverter", "mt5_text2vec_cmn_Hani_32_inverter"]

model_list_corrector_mono = ["mt5_gtr_deu_Latn_32_corrector", "mt5_alephbert_heb_Hebr_32_corrector", "mt5_text2vec_cmn_Hani_32_corrector"]

eval_langs = [
    'deu_Latn', 'mlt_Latn', 'tur_Latn', 'hun_Latn', 'fin_Latn',
    'kaz_Cyrl', 'mhr_Cyrl', 'mon_Cyrl',
    'ydd_Hebr', 'heb_Hebr',
    'arb_Arab', 'urd_Arab',
    'hin_Deva', 'guj_Gujr', 'sin_Sinh', 'pan_Guru',
    'cmn_Hani', 'jpn_Jpan', 'kor_Hang', 'amh_Ethi'
]
