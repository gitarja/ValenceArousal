for /L %%i in (1, 1, 0) do (
   echo %%i  the current iteration
   C:\Users\ShimaLab\Anaconda3\envs\tensorflow\python.exe D:/usr/pras/project/ValenceArousal/TrainPreTrain.py %%i
)
for /L %%i in (1, 1, 0) do (
   echo %%i  the current iteration
   C:\Users\ShimaLab\Anaconda3\envs\tensorflow\python.exe D:/usr/pras/project/ValenceArousal/TrainingStochasticEnsemble.py %%i
)

for /L %%i in (1, 1, 5) do (
   echo %%i  the current iteration
   C:\Users\ShimaLab\Anaconda3\envs\tensorflow\python.exe D:/usr/pras/project/ValenceArousal/TrainingKD.py %%i
)