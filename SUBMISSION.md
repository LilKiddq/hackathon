# Submission: Kuchbrtna

## Links

- Repository: https://github.com/LilKiddq/hackathon
- Final commit: https://github.com/LilKiddq/hackathon/commit/9357ef1c745a6672548ec1b782aabe68e4a85fa2
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

Подтверждённые запуски: lm_head_ce_kernel, nf4_dequant_kernel,
rms_norm_bwd_kernel, rms_norm_fwd_kernel, rope_rotate_kernel.
Суммарная доля: 9,3% по check 4; индивидуальные доли здесь не подтверждены.
SwiGLU — пример организаторов, не собственный вклад.
Алгоритмы и источники: docs/IMPLEMENTATION.md и README.md. Архив исходников совпал с solution_sha256 из проверки.

## Results on Kaggle T4

| Метрика | Base Soup | Решение | Отношение |
|---|---|---|---|
| Полное время, с | 89,56 | 70,31 | 1,2738× ускорение |
| Пик GPU-памяти, MiB | 14 345 | 5 465 | 2,6249× меньше |
| Eval loss | 2,04966 | 2,04966 | практически без изменения |

- [x] Все 10 checks PASS по приложенным evidence/checks.json.
- [x] 79 тестов Soup PASS.
- [x] Техническое описание сверено с архивом исходников; источники из комментариев перечислены. Происхождение кода подтверждает команда.
- [x] Финальный commit подтверждён файлом COMMIT.txt.
- [ ] Подтверждена публичность нового notebook и добавлен scriptVersionId.
- [x] Название команды и четыре участника заполнены.
- [ ] Роли, профили и контакт капитана заполнены.
- [ ] Видео и презентация добавлены.
- [ ] Подтверждено, что репозиторий создан платформой команды и доступен жюри.

## Limitations

Одна пара запусков, seed=1, открытые данные; скрытая проверка жюри ещё не выполнена.
Результаты и хеши: RESULTS.md. Происхождение кода подтверждает команда.
