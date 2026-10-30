# Kuchbrtna — Soup Kernel Challenge

## Реализация и источники идей

Пакет подключается через solution.apply() до создания модели и trainer. Состав батчей,
LoRA, оптимизатор и расписание не меняются.

| Компонент | Файл / кернел | Что реализовано |
|---|---|---|
| Head и CE | fused_ce.py / lm_head_ce_kernel | Logits частями до 512 строк; online log-sum-exp, loss, entropy, argmax и градиент logits в одном кернеле |
| Интеграция | head_patch.py | Метрики TRL и token loss для rewind; замороженный tied embedding/head хранится fp16, embedding output возвращается fp32 |
| RMSNorm | rmsnorm.py / rms_norm_fwd_kernel, rms_norm_bwd_kernel | Слитые forward/backward по строкам, сохранение входа и inverse RMS |
| RoPE | rope.py / rope_rotate_kernel | Split-half rotation q/k и транспонированное преобразование градиента, чтение strided входов без предварительной contiguous-копии |
| NF4 | nf4.py / nf4_dequant_kernel | Double-quantized scale и распаковка NF4 в один запуск, отдельный matmul PyTorch/cuBLAS; поддержан замороженный bias |
| Add + RMSNorm | add_rmsnorm.py / add_rmsnorm_fwd_kernel, add_rmsnorm_bwd_kernel | Residual add и post-attention RMSNorm одним запуском в каждую сторону; сохраняются только s и inverse RMS |
| MLP fp16 | mlp.py | Gate/up и SwiGLU на fp16-пути, сокращение сохранённых активаций; residual остаётся fp32 |
| SwiGLU | swiglu.py | Пример организаторов; не заявляется собственным кернелом |

В актуальном CE градиент hidden вычисляется уже в forward, пока chunk logits находится в памяти.
Градиент сохраняется fp16 с масштабом — степенью двойки; backward умножает его на grad_total / scale
и возвращает в исходный порядок строк. Это заменяет старую схему повторного matmul head в backward.
Матрицы умножаются средствами PyTorch/cuBLAS: собственный Triton-код обрабатывает статистики и градиент logits.

NF4 явно использует округлённое fp32-умножение scale (mul.rn.f32) и enable_fp_fusion=False,
чтобы не объединять умножение и сложение в FMA. Плотный распакованный вес всё ещё создаётся временно.
RoPE также отключает FMA и сохраняет промежуточное округление произведений.

Индивидуальные выигрыши компонентов не измерены отдельными ablation-запусками.

Источники по комментариям реализации:
- RMSNorm: Zhang & Sennrich, https://arxiv.org/abs/1910.07467.
- RoPE: Su et al., RoFormer, https://arxiv.org/abs/2104.09864; split-half layout Qwen2 в transformers 5.16.1.
- NF4 / double quantization: Dettmers et al., QLoRA, https://arxiv.org/abs/2305.14314; формат bitsandbytes, https://github.com/bitsandbytes-foundation/bitsandbytes.
- Online softmax: Milakov & Gimelshein, https://arxiv.org/abs/1805.02867.
- Chunked head/loss: Soup stream_loss_chunk (#1693 в комментариях), https://github.com/MakazhanAlpamys/Soup;
  идея fused linear CE из https://github.com/linkedin/Liger-Kernel, без подключения библиотеки.
- SwiGLU: стартовый пример https://github.com/MakazhanAlpamys/soup-kernel-challenge.

Исходники объявляют собственную реализацию указанных кернелов; независимый аудит происхождения кода не выполнялся.

## Как запустить

Импортировать templates/kaggle_notebook.ipynb в Kaggle; GPU T4 x2, Internet On.
Используется одна карта. REPO_URL = https://github.com/LilKiddq/hackathon.git.
Для воспроизведения нужен финальный commit измеренной реализации (TBD).

```bash
python task/prepare_data.py
python -m pytest solution -q
python harness/run_task.py
python harness/check_rules.py
python harness/score.py runs/result.json
```

Kaggle, результаты сохранённого запуска: https://www.kaggle.com/code/bloovw/notebookca2/output (ссылка пользователя без scriptVersionId; закреплённую ссылку на версию ещё добавить).
Final commit: 9357ef1c745a6672548ec1b782aabe68e4a85fa2. Команда: Kuchbrtna.

Участники: Madiyar Bakhtiyar, Abu Sabytbek, Nariman Bolatuly, Aidarov Bakhytzhan.

## Результаты и ограничения

> Предварительно: замер на коммите 9357ef1, до подключения add_rmsnorm. Таблица будет обновлена финальным прогоном `--repeats 2`.

| Метрика | Base Soup | Решение | Отношение |
|---|---|---|---|
| Полное время, с | 89,56 | 70,31 | 1,2738× ускорение |
| Пик GPU-памяти, MiB | 14 345 | 5 465 | 2,6249× меньше |
| Eval loss | 2,04966 | 2,04966 | практически без изменения |

Все 10 checks PASS; собственные кернелы 9,3%; 79 тестов Soup PASS.
Один запуск каждой стороны, seed=1, открытая eval-выборка. Скрытые данные и повторы жюри не проверены.
Подробности и контрольные суммы: [RESULTS.md](RESULTS.md).
Исходные результаты: [result.json](evidence/result.json), [checks.json](evidence/checks.json).
Форма сдачи: [SUBMISSION.md](SUBMISSION.md).
Лицензии стартового набора: LICENSE и NOTICE; исходная инструкция в docs/STARTER_GUIDE.md.

## Сверка исходников

Папка solution/ в этом пакете побайтово совпадает с контрольной суммой solution_sha256 из checks.json.
Python __pycache__ исключён, исходный код не менялся. Проверена синтаксическая корректность 10 Python-файлов; GPU-тесты локально не перезапускались.
