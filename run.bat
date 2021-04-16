for /L %%i in (1, 1, 5) do (
   python TrainingTeacher.py %%i
)

for /L %%i in (1, 1, 5) do (
   python TrainingFeatures.py %%i
)

