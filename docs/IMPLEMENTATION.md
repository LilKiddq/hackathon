# Kuchbrtna — техническое описание

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
| Удаление паддинга | unpad.py | ~43 % позиций батча — паддинг. Слои работают только с реальными токенами ([1, n, hidden]); attention считается отдельным causal-вызовом SDPA на каждую строку по её реальной длине (строки идут подряд, q/k/v строки — срез, маска не нужна), RoPE берёт cos/sin по позиции каждого токена; dropout LoRA берёт маску для формы с паддингом, так что маски и RNG те же |
| Offload активаций MLP | offload.py | gate/up каждого слоя сразу после SwiGLU копируются во второй CUDA-поток в pinned-память хоста и освобождаются на GPU; в backward подгружаются на слой раньше, параллельно со счётом. Копии точные. ~2 ГБ pinned-памяти закрепляются в фоне во время загрузки модели |
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
