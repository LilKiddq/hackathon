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
| Удаление паддинга | unpad.py | ~43 % позиций батча — паддинг. Слои работают только с реальными токенами ([1, n, hidden]); attention считается отдельным causal-вызовом SDPA на каждую строку по её реальной длине (строки идут подряд, q/k/v строки — срез, маска не нужна), RoPE берёт cos/sin по позиции каждого токена; dropout LoRA берёт маску для формы с паддингом, так что маски и RNG те же. Совпадает с обучением с паддингом до ~1e-5 на CPU-проверке |
| Offload активаций MLP | offload.py | gate/up каждого слоя (самые большие сохраняемые тензоры, ~1 МБ на токен) сразу после SwiGLU копируются во второй CUDA-поток в pinned-память хоста и освобождаются на GPU; в backward подгружаются на слой раньше, параллельно со счётом. Копии точные. ~2 ГБ pinned-памяти закрепляются в фоне во время загрузки модели |
| Компиляция в фоне | warmup.py | apply() запускает каждый кернел один раз на маленьких тензорах в фоновом потоке, пока Soup загружает модель; компиляция с пустого кэша остаётся внутри замера, но не задерживает первые шаги |
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

Kaggle-ноутбук: TBD (публичная сохранённая версия с COMMIT = финальный коммит).
Измеренный код решения: коммит `8494f13` (последующие коммиты меняют только документацию). Команда: Kuchbrtna.

Участники: Madiyar Bakhtiyar, Abu Sabytbek, Nariman Bolatuly, Aidarov Bakhytzhan.

## Результаты и ограничения

Kaggle Tesla T4, `python harness/run_task.py --repeats 2` (как у жюри: база, решение, решение, база;
среднее двух запусков каждой стороны), открытые строки, seed 1, коммит `8494f13`.

| Метрика | Base Soup | Решение | Отношение |
|---|---:|---:|---:|
| Полное время `soup train`, с | 83,1 (79,19 / 87,05) | 57,3 (56,64 / 57,99) | **1,450×** быстрее |
| Пик GPU-памяти (nvidia-smi), MiB | 14 345 | 3 377 | **4,248×** меньше |
| Eval loss на отложенных строках | 2,04969 | 2,04929 | −0,020 % |
| Медианный шаг обучения, с | 0,907 / 1,026 | 0,467 / 0,511 | ~2,0× |
| Обучающий цикл (64 шага), с | 57,27 / 65,62 | 32,31 / 33,39 | ~1,9× |

Время базы между сессиями Kaggle заметно плавает (76–88 с для того же кода), и вторые запуски каждой
стороны медленнее первых (T4 нагревается), поэтому отношение по времени в разных сессиях отличается на
несколько процентов. Память от сессии не зависит.

`check_rules.py`: все 10 проверок PASS в обоих запусках решения; наши кернелы занимают 10,6 % времени
обучения (минимум из двух запусков, шаги 3+); первая точка кривой loss отличается на 0,04 %,
среднее отклонение 0,06 % (допуск 1 % и 3 %). `pytest solution`: 45 passed; регрессионные тесты Soup: 79 passed.

Как менялся результат по ходу работы (по одному запуску, кроме последней строки):

| Версия | Что добавлено | Скорость | Память |
|---|---|---:|---:|
| v1 | fused CE, RMSNorm, NF4 | 1,09–1,13× | 1,70× |
| v2 | MLP на fp16, embedding fp16, RoPE, bias в NF4 | 1,18× | 2,63× |
| v3 | add + RMSNorm | 1,20× | 2,63× |
| v4 | градиент CE в forward, компиляция в фоне | 1,25× | 2,60× |
| v6 | NF4 и RoPE без лишних приведений, `--repeats 2` | 1,30× | 2,64× |
| v7 | удаление паддинга внутри батча, `--repeats 2` | 1,45× | 2,82× |
| v8 | attention по строкам на реальной длине, `--repeats 2` | 1,58× (база 87,7 с) | 3,22× |
| v9 | offload активаций MLP в pinned-память хоста, `--repeats 2` | **1,45×** (база 83,1 с) | **4,25×** |

Ограничения:
- Основное время шага — матричные умножения cuBLAS и attention (cutlass); их мы не заменяли. После удаления
  паддинга матрицы стали меньше, и на T4 они плохо загружают GPU (квантование по волнам), поэтому
  выигрыш от удаления ~43 % токенов составил ~10–15 % шага, а не 43 %. Около 25 с полного времени
  (загрузка и квантование модели, сохранение адаптера) одинаковы у базы и решения.
- Память: из оставшихся ~3,4 ГБ около 1,5 ГБ постоянны (4-битные веса, эмбеддинг fp16, CUDA-контекст);
  остальное — сохранённые для backward входы нормализаций, LoRA и attention. Offload активаций MLP
  стоит ~3 % времени (копирование через PCIe почти полностью перекрыто счётом); пересчёт (recompute)
  мы не включали: он стоил бы ~10 % скорости.
- Замер на открытых строках; на скрытых строках жюри длины батчей и время могут немного отличаться.
- Отдельные ablation-запуски по каждому кернелу не делались; вклад видно по таблице версий выше
  и по микробенчмаркам ниже.

### Микробенчмарки кернелов (Kaggle T4)

`python tools/bench_kernels.py`, медиана 50 замеров, forward + backward (NF4 — только распаковка).

| Кернел | PyTorch, мс | Наш, мс | Ускорение | Память PyTorch → наша, MiB | Max diff |
|---|---:|---:|---:|---:|---:|
| fused lm_head + CE (2048 × 1536 × 151936) | 197,3 | 107,6 | 1,83× | 5 820 → 635 | 4,6e-5 |
| RMSNorm (4 × 512 × 1536, fp32) | 1,47 | 0,47 | 3,11× | 112 → 64 | 1,4e-6 |
| RoPE (q 4×12×512×128, k 4×2×512×128) | 1,77 | 0,69 | 2,57× | 97 → 85 | 0 |
| NF4 dequant 8960 × 1536 | 0,341 | 0,223 | 1,53× | 50 → 49 | 0 |
| NF4 dequant 1536 × 1536 | 0,153 | 0,096 | 1,59× | 22 → 22 | 0 |

Полный вывод: [evidence/bench_kernels.md](evidence/bench_kernels.md). `pytest solution` на T4: 43 passed.

Подробности: [RESULTS.md](RESULTS.md). Описание реализации: [docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md).
Форма сдачи: [SUBMISSION.md](SUBMISSION.md).
Лицензии стартового набора: LICENSE и NOTICE; исходная инструкция в docs/STARTER_GUIDE.md.
