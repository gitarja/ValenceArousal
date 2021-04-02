for /L %%i in (1, 1, 1) do (
   python TrainingTeacher.py %%i
)
for /L %%i in (1, 1, 1) do (
   python TrainingFeatures.py %%i
)

