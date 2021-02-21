for /L %%i in (1, 1, 5) do (
   echo %%i  the current iteration
   C:\Users\ShimaLab\Anaconda3\envs\tensorflow\python.exe D:/usr/pras/project/ValenceArousal/TrainingStochasticEnsemble_MClass.py %%i
)

for /L %%i in (1, 1, 1) do (
   echo %%i  the current iteration
   C:\Users\ShimaLab\Anaconda3\envs\tensorflow\python.exe D:/usr/pras/project/ValenceArousal/TrainingKD_MClass.py %%i
)