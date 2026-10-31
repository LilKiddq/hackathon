# Submission: Kuchbrtna

## Links

- Repository: https://github.com/LilKiddq/hackathon
- Final commit: TBD (измеренный код решения — v33, коммит cda3a4f; финальный коммит добавляет только документацию)
- Kaggle, сохранённый запуск: https://www.kaggle.com/code/bloovw/notebookca2/output
- Ссылка на конкретную версию с scriptVersionId: TBD. Публичность нового ноутбука ещё подтвердить.
- Demo video, до 3 минут: TBD.
- Presentation: TBD; скелет docs/PRESENTATION_OUTLINE.md.

## Team (до 4 человек)

| Имя | Роль | GitHub | Kaggle | Контакт капитана |
|---|---|---|---|---|
| Madiyar Bakhtiyar | TBD | TBD | TBD | TBD |
| Abu Sabytbek | TBD | TBD | TBD | TBD |
| Nariman Bolatuly | TBD | TBD | TBD | TBD |
| Aidarov Bakhytzhan | TBD | TBD | TBD | TBD |

## Our kernels

Подтверждённые запуски: add_rmsnorm_fwd_kernel, add_rmsnorm_bwd_kernel, head_partial_kernel,
int8_rows_dequant_kernel, nf4_dequant_kernel, nf4_unit_dequant_kernel, rms_norm_fwd_kernel, rms_norm_bwd_kernel,
rope_rotate_kernel. Суммарная доля: 13,0 % по check 4 (минимум из двух запусков); индивидуальные доли не измерялись.
SwiGLU — пример организаторов, не собственный вклад.
Алгоритмы и источники: docs/IMPLEMENTATION.md и README.md.

## Results on Kaggle T4

`harness/run_task.py --repeats 2` (база, решение, решение, база), код решения v33 (cda3a4f):

| Метрика | Base Soup | Решение | Отношение |
|---|---:|---:|---:|
| Полное время, с | 78,0 | 50,7 | 1,539× ускорение |
| Пик GPU-памяти, MiB | 14 345 | 735 | 19,517× меньше |
| Eval loss | 2,04965 | 2,04922 | −0,021 % |

- [x] Все 10 checks PASS (--repeats 2), 57 тестов solution и 79 тестов Soup PASS.
- [x] Техническое описание сверено с архивом исходников; источники из комментариев перечислены. Происхождение кода подтверждает команда.
- [ ] Финальный коммит вписан.
- [ ] Подтверждена публичность нового notebook и добавлен scriptVersionId.
- [x] Название команды и четыре участника заполнены.
- [ ] Роли, профили и контакт капитана заполнены.
- [ ] Видео и презентация добавлены.
- [ ] Подтверждено, что репозиторий создан платформой команды и доступен жюри.

## Limitations

Открытые данные, seed 1; скрытая проверка жюри ещё не выполнена. Подробности и ограничения: README.md и RESULTS.md.
