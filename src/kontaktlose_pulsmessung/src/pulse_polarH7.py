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


def main(addr=None, gatttool="gatttool"):
    """
    main routine to which orchestrates everything
    """
    # set up ROS publisher and node
    pub = rospy.Publisher('pulsgurt', pulse, queue_size=10)
    rospy.init_node('pulsgurt_node', anonymous=False)
    # number of measured pulse values. Increments for every measured value
    seq = 0
    # message to be published is from type pulse. Can be found in pulse.msg
    msg_to_publish = pulse()

    if addr is None:
        # A mac address has to be provided as command line argument
        log.error("MAC address of polar H7 has not been provided")
        return

    hr_handle = None
    hr_ctl_handle = None
    retry = True
    while retry:

        while 1:
            print("Establishing connection to " + addr)
            gt = pexpect.spawn(gatttool + " -b " + addr + " -I")

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


            print("Heart rate: " + str(res["hr"]))
            msg_to_publish.pulse = res["hr"]
            msg_to_publish.time.stamp = rospy.Time.now()
            msg_to_publish.time.seq = seq
            pub.publish(msg_to_publish)
            seq += 1

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
        main(addr=args.m, gatttool=args.g)
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    sys.argv = rospy.myargv()
    cli()
