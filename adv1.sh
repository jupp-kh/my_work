
##Then run this
python adv1_collision_attack.py --save_examples 10 --projection linf --disable_progress_bar --epsilon 0.1 --learning_rate 5
#l2 mean successful: 6.100284
# l_inf mean successful: 0.105558
# steps mean successful: 79.111778
# ===>Collision Rate under l2=11.000001: 95.00999999999999%

#under l2 norm
python adv1_collision_attack.py --save_examples 10 --projection l2 --disable_progress_bar --epsilon 11

