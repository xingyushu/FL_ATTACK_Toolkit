#!/bin/bash

# COUNTER=0

# mkdir log

# for DATASET in "MNIST" "Fashion-MNIST" "CIFAR-10"
# do
# 	for ATTACK in  "krum"  "trimmedmean"
# 	do
# 		for AGG in    "median"  "trimmedmean" "clustering" "bulyankrum"
# 		do
# 		  	for METHOD in   "PlainCSI","FL_TDoS","FL_CPE","fix-1","fix-2"
# 			do

# 				DEVICE=$((COUNTER%8))
# 				COMMAND=" python3 src/simulate.py --dataset ${DATASET} --device ${DEVICE} --attack ${ATTACK} --agg ${AGG} --method ${METHOD} > log/${ATTACK}_${AGG}_${DATASET}_${METHOD}.log;"
# 				echo $COMMAND
# 				screen -dmS ${ATTACK}_${AGG}_${DATASET} bash -c " python3 src/simulate.py --dataset ${DATASET} --device ${DEVICE} --attack ${ATTACK} --agg ${AGG} --method ${METHOD} > log/${ATTACK}_${AGG}_${DATASET}.log;"
# 				COUNTER=$((COUNTER+1))
# 		done
# 	done
# done


#!/bin/bash

COUNTER=0

mkdir -p log_final  # Create log directory if it doesn't exist

for DATASET in "MNIST" "Fashion-MNIST" "CIFAR-10"
do
    for ATTACK in "krum" "trimmedmean"
    do
        for AGG in "median" "trimmedmean" "clustering" "bulyankrum"
        do
            for METHOD in "PlainCSI" "FL_TDoS" "fix-1" "fix-2" "FL_CPE"
            do
                LOG_FILE="log_final/${ATTACK}_${AGG}_${DATASET}_${METHOD}.log"
                COMMAND="python3 src/simulate.py --dataset ${DATASET} --attack ${ATTACK} --agg ${AGG} --method ${METHOD} > ${LOG_FILE} 2>&1"
                echo $COMMAND
                eval ${COMMAND}
            done
        done
    done
done



# #!/bin/bash

# COUNTER=0

# mkdir -p log  # Create log directory if it doesn't exist

# for DATASET in "MNIST" "Fashion-MNIST" "CIFAR-10"
# do
#     for ATTACK in "krum" "trimmedmean"
#     do
#         for AGG in "median" "trimmedmean" "clustering" "bulyankrum"
#         do
#             for METHOD in "PlainCSI" "FL_TDoS" "FL_CPE" "fix-1" "fix-2"
#             do
#                 DEVICE=$((COUNTER%8))
#                 LOG_FILE="log/${ATTACK}_${AGG}_${DATASET}_${METHOD}.log"
#                 COMMAND="python3 src/simulate.py --dataset ${DATASET} --device ${DEVICE} --attack ${ATTACK} --agg ${AGG} --method ${METHOD}"
#                 echo "Running: $COMMAND"
#                 $COMMAND 2>&1 | tee -a ${LOG_FILE}
#                 COUNTER=$((COUNTER+1))
#             done
#         done
#     done
# done
