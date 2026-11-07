# Member 3 integration

هذه إضافة جزئيتك إلى هيكل المشروع الحالي. لا تغيّر كود العضو الثاني أو تقسيم المرضى أو إعدادات المعالجة.

## الملفات

- `src/module1_segmentation/model/unet3d.py`: معمارية MONAI UNet بأوزان عشوائية.
- `src/module1_segmentation/model/losses.py`: Dice + Cross-Entropy مع واجهة القناع المتفق عليها.
- `scripts/check_member3.py`: يربط DataLoader العضو الثاني بالنموذج والخسارة.
- `requirements-member3.txt`: متطلبات إضافية. يستخدم كود العضو الثاني NumPy ومتطلبات المشروع الأصلية.

كودا النموذج والخسارة مطابقان لنسختك السابقة؛ التعديل هو وضعهما في المسارات الصحيحة وإضافة اختبار الربط.

## التشغيل من جذر المشروع

ضع محتويات هذه الإضافة داخل مجلد المشروع مع الحفاظ على مساراتها. تستبدل ملفي النموذج والخسارة اللذين يحتويان TODO فقط في النسخة المرفقة. إذا تغيرا لاحقًا في GitHub، قارن التعديلات قبل النسخ. يفضل العمل في فرع member3-model-integration ورفع Pull Request بعد الفحص.

ثبت المكتبات مع إبقاء إصدار PyTorch المتوافق مع GPU لديك:

```bash
python -m pip install -r requirements.txt -r requirements-member3.txt
```

اختبار التكامل الصناعي بحجم 96 عبر DataLoader العضو الثاني:

```bash
python scripts/check_member3.py --synthetic --patch-size 96 96 96 --batch-size 1 --device cpu
```

هذا يخلق ملفات صناعية مؤقتة ويمررها إلى نفس Dataset/DataLoader ثم النموذج والخسارة والمرور الخلفي. لا يثبت صحة أشعة المرضى أو جودة التجزئة.

لتجربة البيانات الفعلية، شغله في جهاز يحتوي الكاش المحلي للعضو الثاني، أو انسخ حالات مجهزة مع الأقنعة والمراكز والـmetadata وقوائم CSV التي تشير إليها. اختبار التكامل الافتراضي يقرأ قوائم train/val كاملة، لذا إن اختبرت حالتين فقط، أنشئ مجلد قوائم منفصلًا يحوي `train.csv` و`val.csv` برأس CSV الأصلي وصف واحد لكل حالة، دون تعديل القوائم الأصلية. يجب أن تكون ملفات كل حالة مدرجة في القوائم موجودة محليًا. استخدم ملفات من الأقسام المحددة نفسها وبصمة معالجة واحدة.

```bash
python scripts/check_member3.py --patch-size 96 96 96 --batch-size 1 --device cuda --output outputs/m3_integration/real_batch.json
```

على جهاز بدون CUDA استخدم `--device cpu`. لا يوجد انتقال تلقائي من CUDA إلى CPU.

إن استخدمت قوائم العينتين المنفصلة، أضف `--manifest-dir PATH_TO_SAMPLE_MANIFESTS`.

النتيجة المتوقعة PASS مع data_kind=real_prepared. يختبر الأمر دفعة تدريب ودفعة تحقق؛ يتحقق من الدخل والتسميات والخرج والخسارة والتدرجات. لا يحدث الأوزان ولا يقوم بتدريب فعلي أو اختبار دقة. لا يقرأ test.

## الربط في كود تدريب العضو الرابع

```python
from src.module1_segmentation.model.unet3d import VertebraUNet
from src.module1_segmentation.model.losses import SegmentationLoss

model = VertebraUNet().to(device)
criterion = SegmentationLoss().to(device)
# من DataLoader العضو الثاني: قاموس وليس tuple
images = batch['image'].to(device)
masks = batch['mask'].to(device)
logits = model(images)
loss = criterion(logits, masks)
```

يضيف العضو الرابع optimizer وzero_grad وbackward وstep والحفظ ودورة التدريب. يجب أن يستدعي train_dataset.set_epoch(epoch) قبل كل epoch. الإعداد المبدئي batch_size=1؛ DataLoader العضو الثاني افتراضيًا يستخدم 2، لذلك مرره صراحة لتجارب الدمج.

## ملاحظات المراجعة

واجهة العضوين متوافقة: images float32 [B,1,D,H,W]، masks int64 [B,D,H,W]، labels 0..8. spacing المعتمد في إعدادات العضو الثاني 2 mm على المحاور الثلاثة، وحجم الجزء 96 يغطي طولًا اسميًا 192 mm لكل محور. تبقى هذه الإعدادات مبدئية، وملاءمتها لتحديد مستويات الفقرات تحتاج تقييمًا على البيانات الحقيقية.

القناع يدمج الفقرات خارج T10-L5 ضمن الفئة 0 باعتبارها non-target؛ الحفاظ على الاتفاق مطلوب. الصور غير المعدلة محفوظة لدى العضو الثاني لاستخدام HU لاحقًا.

ملف GitHub المرفق يحتوي قوائم 50 تدريب و69 تحقق و60 اختبار، لكنه لا يحتوي ملفات .npy أو صور CT. ادعاء اكتمال تجهيز البيانات في توثيق العضو الثاني لا يمكن تأكيده من هذه النسخة وحدها. تقرير هذه الإضافة صناعي وموسوم بذلك صراحة.

المراجعة البصرية لمحاذاة الصور والأقنعة، واختبار الاستدلال الكامل وإعادة الإحداثيات، وتقييم Dice على المرضى، أعمال منفصلة لم ينجزها اختبار الربط هذا.
