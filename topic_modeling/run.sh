for p in `ps -elf | grep python | awk '{ print $4 }'`
do
sudo kill -9 $p
done

# python GSM_run.py --taskname ses_anxi --n_topic 10 --batch_size 16 --num_epochs 100 --no_above 0.3 --no_below 3 --criterion cross_entropy > out/anxi_nvdmgsm.out &
# python GSM_run.py --taskname ses_depr --n_topic 10 --batch_size 16 --num_epochs 100 --no_above 0.3 --no_below 3 --criterion cross_entropy > out/depr_nvdmgsm.out &
# python GSM_run.py --taskname ses_schi --n_topic 10 --batch_size 16 --num_epochs 100 --no_above 0.3 --no_below 3 --criterion cross_entropy > out/schi_nvdmgsm.out &

# python WTM_run.py --taskname ses_anxi --n_topic 10 --batch_size 16 --num_epochs 100 --no_above 0.3 --dist dirichlet > out/anxi_wtmmmd.out
# python WTM_run.py --taskname ses_depr --n_topic 10 --batch_size 16 --num_epochs 100 --no_above 0.3 --dist dirichlet > out/depr_wtmmmd.out 
# python WTM_run.py --taskname ses_schi --n_topic 10 --batch_size 16 --num_epochs 100 --no_above 0.3 --dist dirichlet > out/schi_wtmmmd.out 

# python WTM_run.py --taskname ses_anxi --n_topic 10 --batch_size 16 --num_epochs 100 --dist gmm-ctm --no_below 3 --auto_adj > out/anxi_wtmgmm.out 
# python WTM_run.py --taskname ses_depr --n_topic 10 --batch_size 16 --num_epochs 100 --dist gmm-ctm --no_below 3 --auto_adj > out/depr_wtmgmm.out 
# python WTM_run.py --taskname ses_schi --n_topic 10 --batch_size 16 --num_epochs 100 --dist gmm-ctm --no_below 3 --auto_adj > out/schi_wtmgmm.out 

# python ETM_run.py --taskname ses_anxi --n_topic 10 --batch_size 16 --num_epochs 100 --no_below 3 --auto_adj --emb_dim 300 > out/anxi_etm.out &
# python ETM_run.py --taskname ses_depr --n_topic 10 --batch_size 16 --num_epochs 100 --no_below 5 --auto_adj --emb_dim 300 > out/depr_etm.out &
# python ETM_run.py --taskname ses_schi --n_topic 10 --batch_size 16 --num_epochs 100 --no_below 3 --auto_adj --emb_dim 300 > out/schi_etm.out &

# python GMNTM_run.py --taskname ses_anxi --n_topic 10 --batch_size 16 --num_epochs 100 --no_below 3 --auto_adj > out/anxi_gmntm.out &
# python GMNTM_run.py --taskname ses_depr --n_topic 10 --batch_size 16 --num_epochs 100 --no_below 5 --auto_adj > out/depr_gmntm.out &
# python GMNTM_run.py --taskname ses_schi --n_topic 10 --batch_size 16 --num_epochs 100 --no_below 3 --auto_adj > out/schi_gmntm.out &

# python BATM_run.py --taskname ses_anxi --n_topic 10 --batch_size 16 --num_epochs 100 --no_above 0.3 --no_below 3 > out/anxi_batm.out &
# python BATM_run.py --taskname ses_depr --n_topic 10 --batch_size 16 --num_epochs 100 --no_above 0.3 --no_below 3 > out/depr_batm.out &
# python BATM_run.py --taskname ses_schi --n_topic 10 --batch_size 16 --num_epochs 100 --no_above 0.3 --no_below 3 > out/schi_batm.out &

# python GSM_run.py --taskname ses_suic --n_topic 10 --batch_size 16 --num_epochs 100 --no_above 0.8 --no_below 3 --criterion cross_entropy > out/suic_nvdmgsm.out &
# python WTM_run.py --taskname ses_suic --n_topic 10 --batch_size 16 --num_epochs 100 --no_above 0.3 --dist dirichlet > out/suic_wtmmmd.out &
# python WTM_run.py --taskname ses_suic --n_topic 10 --batch_size 16 --num_epochs 100 --dist gmm-ctm --no_below 3 --auto_adj > out/suic_wtmgmm.out &
# python ETM_run.py --taskname ses_suic --n_topic 10 --batch_size 16 --num_epochs 100 --no_below 3 --auto_adj --emb_dim 300 > out/suic_etm.out &
# python GMNTM_run.py --taskname ses_suic --n_topic 10 --batch_size 16 --num_epochs 100 --no_below 3 --auto_adj > out/suic_gmntm.out &
# python BATM_run.py --taskname ses_suic --n_topic 10 --batch_size 16 --num_epochs 100 --no_above 0.3 --no_below 3 > out/suic_batm.out &

python ETM_run.py --taskname ses_all --n_topic 10 --batch_size 16 --num_epochs 100 --no_below 3 --auto_adj --emb_dim 300 > out/all_etm.out &
