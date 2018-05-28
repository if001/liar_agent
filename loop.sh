#!/bin/bash

bar(){ #プログレスバー
    now_progress=$1
    max_progress=$2
    bar=("[          ]" "[-         ]" "[--        ]" "[---       ]" "[----      ]" "[-----     ]" "[------    ]" "[-------   ]" "[--------  ]" "[--------- ]" "[----------]")
    num=`expr ${now_progress} \* 100 / ${max_progress}`
    bnum=`expr ${num}/10`
    echo -e "${bar[${bnum}]}"${num}"%\r\c"

    if [ $now_progress == $max_progress ] ;then
	echo  -e "\c"
    fi
}


LOOP=1000
for i in `seq 0 $LOOP`; do
    bar $i $LOOP
    python3 main.py
done

