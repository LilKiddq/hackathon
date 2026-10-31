# Submission: Lean Soup (команда Kuchbrtna)

- Project: **Lean Soup — дообучение Qwen2.5-1.5B на 735 МБ видеопамяти вместо 14,3 ГБ, в 1,5 раза быстрее**

## Links

- Repository: https://github.com/BAITC-Hacks/hack-0e0f2cfa-kuchbrtna (репозиторий команды от организаторов; до 31.10
  разработка шла в https://github.com/LilKiddq/hackathon, вся история перенесена без изменений)
- Final commit: 28b3a68 (код решения — v34, коммит 6b32d62; последующие коммиты меняют только документацию)
- Публичный прогон Kaggle (клон репозитория команды, коммит 28b3a68, `--repeats 2`): база 77,8 с, решение 51,1 с,
  **1,522×**; 14 345 → 735 MiB, **19,517×**; eval loss −0,017 %; ALL CHECKS PASSED; 57 passed
- Kaggle, публичный ноутбук финального прогона: https://www.kaggle.com/code/narimanbolatuly/notebook0627412735
- Demo video: https://youtu.be/9l_tnI4UKt0
- Presentation: https://drive.google.com/file/d/1UOJRMKyuMqFC15Ceu5jvA5J6Dvp9whHT/view?usp=sharing

## Team (до 4 человек)

| Имя | Роль | GitHub | Kaggle | Контакт капитана |
|---|---|---|---|---|
| Nariman Bolatuly | Капитан проекта | [bayesdz](https://github.com/bayesdz) | [narimanbolatuly](https://www.kaggle.com/narimanbolatuly) | Telegram @bratiko1, 11townhall321@gmail.com |
| Aidarov Bakhytzhan | Участник | [LilKiddq](https://github.com/LilKiddq) | [exagee](https://www.kaggle.com/exagee) | — |
| Abu Sabytbek | Участник | [strogenyakuzo-blip](https://github.com/strogenyakuzo-blip) | [bloovw](https://www.kaggle.com/bloovw) | — |
| Madiyar Bakhtiyar | Участник; капитан при регистрации на edu.astanahub.com | [Mad1yarBakhtiyar](https://github.com/Mad1yarBakhtiyar) | [madiyarbakhtiyar](https://www.kaggle.com/madiyarbakhtiyar) | — |

Все прогоны на Kaggle выполнялись с аккаунта narimanbolatuly.

## Our kernels

Подтверждённые запуски: add_rmsnorm_fwd_kernel, add_rmsnorm_bwd_kernel, head_partial_kernel,
int8_rows_dequant_kernel, nf4_unit_dequant_kernel, rms_norm_fwd_kernel, rms_norm_bwd_kernel,
rope_rotate_kernel. Суммарная доля: 13,2 % по check 4 (минимум из двух запусков); индивидуальные доли не измерялись.
SwiGLU — пример организаторов, не собственный вклад.
Алгоритмы и источники: docs/IMPLEMENTATION.md и README.md.

## Results on Kaggle T4

`harness/run_task.py --repeats 2` (база, решение, решение, база), код решения v34 (6b32d62):

| Метрика | Base Soup | Решение | Отношение |
|---|---:|---:|---:|
| Полное время, с | 77,8 | 50,2 | 1,548× ускорение |
| Пик GPU-памяти, MiB | 14 345 | 735 | 19,517× меньше |
| Eval loss | 2,04969 | 2,04928 | −0,020 % |

- [x] Все 10 checks PASS (--repeats 2), 57 тестов solution и 79 тестов Soup PASS.
- [x] Техническое описание сверено с архивом исходников; источники из комментариев перечислены. Происхождение кода подтверждает команда.
- [x] Финальный коммит вписан (28b3a68).
- [x] Публичный Kaggle-ноутбук финального прогона добавлен.
- [x] Название команды и четыре участника заполнены.
- [x] Роли и профили заполнены.
- [x] Видео и презентация добавлены.
- [x] Код в репозитории команды от организаторов (BAITC-Hacks/hack-0e0f2cfa-kuchbrtna), история перенесена целиком.

## Limitations

Открытые данные, seed 1; скрытая проверка жюри ещё не выполнена. Подробности и ограничения: README.md и RESULTS.md.
