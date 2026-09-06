for ip in 159.93.225.202 159.93.225.204 159.93.225.212 159.93.225.213; do
    echo -n "$ip: "; curl -s --connect-timeout 5 telnet://$ip:22 2>/dev/null | head -1 || echo "timeout"
done
