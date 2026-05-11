#!/bin/bash
folder_name="./raw_data"
new_folder_root="./csv_data"
new_folder_path=${new_folder_root}
file_name="./raw_data/N3_Facebook_129_0910_094907.pcap"
output_file="./csv_data/N3_Facebook_129_0910_094907.csv"

# mkdir ${new_folder_path}

# for file_name in ${folder_name}/*
# do
# 	tmp=${file_name##*$folder_name}
# 	output_file=${new_folder_path}${tmp%%pcap*}csv
# 	echo ${output_file}
# 	tshark -r ${file_name} -2 -T fields -e frame.time_epoch -e frame.time_relative -e frame.len -e ip.src -e ip.dst -e ip.proto -e ip.len -e ip.hdr_len -e tcp.srcport -e tcp.dstport -e tcp.len -e tcp.hdr_len -e tcp.flags.syn -e tcp.flags.ack -e tcp.flags.fin -e tcp.flags.urg -e tcp.flags.push -e tcp.flags.reset -e tcp.analysis.out_of_order -e tcp.analysis.retransmission -e tcp.window_size_value -e tcp.analysis.ack_rtt -e tcp.analysis.bytes_in_flight -e tcp.seq -e tcp.ack -e udp.srcport -e udp.dstport -e udp.length -E header=y -E separator=',' -E occurrence=f -E header=y -E bom=y > ${output_file}
# done

for file_name in ${folder_name}/*
do
	tmp=${file_name##*$folder_name}
	output_file=${new_folder_path}${tmp%%pcap*}csv
	echo ${output_file}
	tshark -r ${file_name} \
			-Y "gtp" \
			-T fields \
			-E header=y -E separator="," -E quote=d -E occurrence=a -E aggregator=";" \
			-e frame.time_epoch -e frame.time_relative -e frame.len \
			-e ip.src -e ip.dst -e ip.proto -e ip.len -e ip.hdr_len \
			-e ipv6.src -e ipv6.dst -e ipv6.plen \
			-e tcp.srcport -e tcp.dstport -e tcp.len -e tcp.hdr_len -e tcp.flags.syn -e tcp.flags.ack \-e tcp.flags.fin -e tcp.flags.urg -e tcp.flags.push -e tcp.flags.reset -e tcp.analysis.out_of_order -e tcp.analysis.retransmission -e tcp.window_size_value -e tcp.analysis.ack_rtt -e tcp.analysis.bytes_in_flight -e tcp.seq -e tcp.ack -e tcp.analysis.duplicate_ack -e tcp.analysis.spurious_retransmission -e tcp.analysis.lost_segment -e tcp.analysis.ack_lost_segment \
			-e udp.srcport -e udp.dstport -e udp.length \
			-e icmp.type -e icmp.code \
			-e quic.spin_bit -e quic.header_form \
			-e quic.long.packet_type \
			> ${output_file}
done
