job_name="imagenet"
local_proc_num=4

echo 'Process killed: '
ps -ef | grep $job_name | head -n $local_proc_num
ps -ef | grep $job_name | head -n $local_proc_num | awk '{print $2}' | xargs kill

#main_pid=$(pgrep -f "job_name $job_name")
#if [ -n "$main_pid" ]; then
#    kill "$main_pid"
#    echo "Main process with PID $main_pid killed."
#else
#    echo "No main process found with job name $job_name."
#fi