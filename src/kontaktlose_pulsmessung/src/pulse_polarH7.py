#!/usr/bin/python3
# -*- encoding: utf-8 -*-
"""
    BLEHeartRateLogger
    ~~~~~~~~~~~~~~~~~~~

    A tool to log your heart rate using a Bluetooth low-energy (BLE) heart rate
    monitor (HRM). The tool uses system commands (hcitool and gatttool) to
    connect to the BLE HRM and parses the output of the tools. Data is
    interpreted according to the Bluetooth specification for HRM and saved in a
    sqlite database for future processing. In case the connection with the BLE
    HRM is lost, connection is restablished.

    :copyright: (c) 2015 by fg1
    :license: BSD, see LICENSE for more details
"""

__version__ = "0.1.1"

import os
import sys
import time
import logging
import sqlite3
import pexpect
import argparse
import configparser
import rospy
from kontaktlose_pulsmessung.msg import pulse
from std_msgs.msg import String
logging.basicConfig(format="%(asctime)-15s  %(message)s")
log = logging.getLogger("BLEHeartRateLogger")


def parse_args():
    """
    Command line argument parsing
    """
    parser = argparse.ArgumentParser(description="Bluetooth heart rate monitor data logger")
    parser.add_argument("-m", metavar='MAC', type=str, help="MAC address of BLE device (default: auto-discovery)")
    parser.add_argument("-g", metavar='PATH', type=str, help="gatttool path (default: system available)", default="gatttool")
    parser.add_argument("-H", metavar='HR_HANDLE', type=str, help="Gatttool handle used for HR notifications (default: none)")

    return parser.parse_args()


def interpret(data):
    """
    data is a list of integers corresponding to readings from the BLE HR monitor
    """

    byte0 = data[0]
    res = {}
    res["hrv_uint8"] = (byte0 & 1) == 0
    sensor_contact = (byte0 >> 1) & 3
    if sensor_contact == 2:
        res["sensor_contact"] = "No contact detected"
    elif sensor_contact == 3:
        res["sensor_contact"] = "Contact detected"
    else:
        res["sensor_contact"] = "Sensor contact not supported"
    res["ee_status"] = ((byte0 >> 3) & 1) == 1
    res["rr_interval"] = ((byte0 >> 4) & 1) == 1

    if res["hrv_uint8"]:
        res["hr"] = data[1]
        i = 2
    else:
        res["hr"] = (data[2] << 8) | data[1]
        i = 3

    if res["ee_status"]:
        res["ee"] = (data[i + 1] << 8) | data[i]
        i += 2

    if res["rr_interval"]:
        res["rr"] = []
        while i < len(data):
            # Note: Need to divide the value by 1024 to get in seconds
            res["rr"].append((data[i + 1] << 8) | data[i])
            i += 2

    return res


def insert_db(sq, res, period, min_ce=2, max_ce=60 * 2, grace_commit=2 / 3.):
    """
    Inserts data into the database
    """

    if not hasattr(insert_db, "i"):
        insert_db.i = 0
    if not hasattr(insert_db, "commit_every"):
        insert_db.commit_every = 5

    tstamp = int(time.time())
    if res.has_key("rr"):
        for rr in res["rr"]:
            sq.execute("INSERT INTO hrm VALUES (?, ?, ?)", (tstamp, res["hr"], rr))
    else:
        sq.execute("INSERT INTO hrm VALUES (?, ?, ?)", (tstamp, res["hr"], -1))

    # Instead of pushing the data to the disk each time, we commit only every
    # 'commit_every'.
    if insert_db.i < insert_db.commit_every:
        insert_db.i += 1
    else:
        t = time.time()
        sq.commit()
        delta_t = time.time() - t
        log.debug("sqlite commit time: " + str(delta_t))
        sq.execute("INSERT INTO sql VALUES (?, ?, ?)", (int(t), delta_t, insert_db.commit_every))

        # Because the time for commiting to the disk is not known in advance,
        # we measure it and automatically adjust automatically 'commit_every'
        # following a rule similar to TCP Reno.
        if delta_t < period * grace_commit:
            insert_db.commit_every = min(insert_db.commit_every + 1, max_ce)
        else:
            insert_db.commit_every = max(insert_db.commit_every / 2, min_ce)

        insert_db.i = 0


def get_ble_hr_mac():
    """
    Scans BLE devices and returs the address of the first device found.
    """

    while 1:
        print("Trying to find a BLE device")
        hci = pexpect.spawn("hcitool lescan")
        try:
            hci.expect("([0-9A-F]{2}[:-]){5}([0-9A-F]{2})", timeout=20)
            addr = hci.match.group(0)
            hci.close()
            break

        except pexpect.TIMEOUT:
            time.sleep(20)
            continue

        except KeyboardInterrupt:
            print("Received keyboard interrupt. Quitting cleanly.")
            hci.close()
            return None

    # We wait for the 'hcitool lescan' to finish
    time.sleep(1)
    return addr


def main(addr=None, sqlfile=None, gatttool="gatttool", check_battery=False, hr_handle=None, debug_gatttool=False):
    """
    main routine to which orchestrates everything
    """
    pub = rospy.Publisher('pulsgurt', pulse, queue_size=10)
    rospy.init_node('pulsgurt_node', anonymous=False)
    seq = 0
    # rospy.sleep(1)
    msg_to_publish = pulse()
    rospy.sleep(1)
    if sqlfile is not None:
        # Init database connection
        sq = sqlite3.connect(sqlfile)
        with sq:
            sq.execute("CREATE TABLE IF NOT EXISTS hrm (tstamp INTEGER, hr INTEGER, rr INTEGER)")
            sq.execute("CREATE TABLE IF NOT EXISTS sql (tstamp INTEGER, commit_time REAL, commit_every INTEGER)")

    if addr is None:
        # In case no address has been provided, we scan to find any BLE devices
        addr = get_ble_hr_mac()
        if addr == None:
            sq.close()
            return

    hr_ctl_handle = None
    retry = True
    while retry:

        while 1:
            print("Establishing connection to " + addr)
            gt = pexpect.spawn(gatttool + " -b " + addr + " -I")
            if debug_gatttool:
                gt.logfile = sys.stdout

            gt.expect(r"\[LE\]>")
            gt.sendline("connect")
            try:
                i = gt.expect(["Connection successful.", r"\[CON\]"], timeout=30)
                if i == 0:
                    gt.expect(r"\[LE\]>", timeout=30)

            except pexpect.TIMEOUT:
                print("Connection timeout. Retrying.")
                continue

            except KeyboardInterrupt:
                print("Received keyboard interrupt. Quitting cleanly.")
                retry = False
                break
            break

        if not retry:
            break

        print("Connected to " + addr)

        if check_battery:
            gt.sendline("char-read-uuid 00002a19-0000-1000-8000-00805f9b34fb")
            try:
                gt.expect("value: ([0-9a-f]+)")
                battery_level = gt.match.group(1)
                print("Battery level: " + str(int(battery_level, 16)))

            except pexpect.TIMEOUT:
                log.error("Couldn't read battery level.")

        if hr_handle == None:
            # We determine which handle we should read for getting the heart rate
            # measurement characteristic.
            gt.sendline("char-desc")

            while 1:
                try:
                    gt.expect(r"handle: (0x[0-9a-f]+), uuid: ([0-9a-f]{8})", timeout=10)
                except pexpect.TIMEOUT:
                    break
                handle = gt.match.group(1)
                uuid = gt.match.group(2)

                if uuid == b"00002902" and hr_handle:
                    hr_ctl_handle = handle
                    break

                elif uuid == b"00002a37":
                    hr_handle = handle

            if hr_handle == None:
                log.error("Couldn't find the heart rate measurement handle?!")
                return

        if hr_ctl_handle:
            # We send the request to get HRM notifications
            gt.sendline("char-write-req " + hr_ctl_handle.decode("utf-8") + " 0100")

        # Time period between two measures. This will be updated automatically.
        period = 1.
        last_measure = time.time() - period
        hr_expect = "Notification handle = " + hr_handle.decode("utf-8") + " value: ([0-9a-f ]+)"

        while 1:
            try:
                gt.expect(hr_expect, timeout=10)

            except pexpect.TIMEOUT:
                # If the timer expires, it means that we have lost the
                # connection with the HR monitor
                log.warn("Connection lost with " + addr + ". Reconnecting.")
                if sqlfile is not None:
                    sq.commit()
                gt.sendline("quit")
                try:
                    gt.wait()
                except:
                    pass
                time.sleep(1)
                break

            except KeyboardInterrupt:
                print("Received keyboard interrupt. Quitting cleanly.")
                retry = False
                break

            # We measure here the time between two measures. As the sensor
            # sometimes sends a small burst, we have a simple low-pass filter
            # to smooth the measure.
            tmeasure = time.time()
            period = period + 1 / 16. * ((tmeasure - last_measure) - period)
            last_measure = tmeasure

            # Get data from gatttool
            datahex = gt.match.group(1).strip()
            data = map(lambda x: int(x, 16), datahex.split(b' '))
            data = list(data)
            res = interpret(data)

            log.debug(res)

            if sqlfile is None:
                print("Heart rate: " + str(res["hr"]))
                msg_to_publish.pulse = res["hr"]
                msg_to_publish.time.stamp = rospy.Time.now()
                msg_to_publish.time.seq = seq
                pub.publish(msg_to_publish)
                seq += 1
                continue

            # Push the data to the database
            insert_db(sq, res, period)

    if sqlfile is not None:
        # We close the database properly
        sq.commit()
        sq.close()

    # We quit close the BLE connection properly
    gt.sendline("quit")
    try:
        gt.wait()
    except:
        pass


def cli():
    """
    Entry point for the command line interface
    """
    args = parse_args()

    if args.g != "gatttool" and not os.path.exists(args.g):
        log.critical("Couldn't find gatttool path!")
        sys.exit(1)

    try:
        main(addr=args.m, gatttool=args.g, hr_handle=args.H)
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    sys.argv = rospy.myargv()
    cli()