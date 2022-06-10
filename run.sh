batch_size=8

# for lr in 2e-2
# do
#     for epochs in 20
#     do
#         for lambd in 1e-7
#         do
#             for momentum in 0.8 0.7 0.6 0.5 0.4 0.3
#             do
#                 for temp_ensem_weight in 0.5 1.0
#                 do
#                     python run.py --batch_size ${batch_size} --lr ${lr} --lambd ${lambd} --epochs ${epochs} --add_new_inst 1 --momentum ${momentum} --temp_ensem_weight ${temp_ensem_weight}
#                 done
#             done
#         done
#     done
# done

for lr in 1e-2
do
    for epochs in 30
    do
        for lambd in 1e-7
        do
            for sample_num in 5
            do
                python run.py --batch_size ${batch_size} --lr ${lr} --lambd ${lambd} --epochs ${epochs} --add_new_inst 1 --unmasker_new_model 1 --save_res "update_unmasker" --sample_num ${sample_num}
            done
        done
    done
done



# for lr in 2e-5 #5e-5 1e-5
# do
#     for epochs in 20
#     do
#         for lambd in 1
#         do
#             python run.py --batch_size ${batch_size} --lr ${lr} --lambd ${lambd} --epochs ${epochs}
#         done
#     done
# done


