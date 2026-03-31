/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
package com.arm.ai_ml_sdk_scenario_runner;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.Service;
import android.content.Intent;
import android.os.IBinder;
import android.util.Log;

public class Main extends Service {
    private static final String TAG = "ScenarioService";
    private static final String CHANNEL_ID = "scenario_runner";

    static { System.loadLibrary("scenario_runner_jni"); }

    private native int runScenarioRunner(String[] args);

    @Override
    public void onCreate() {
        super.onCreate();
        createNotificationChannel();
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        startForeground(1, buildNotification());

        new Thread(() -> {
            try {
                String[] extraArgs = intent != null ? intent.getStringArrayExtra("args") : null;

                String[] args = extraArgs != null ? extraArgs : new String[0];

                if (extraArgs != null) {
                    for (String arg : extraArgs) {
                        Log.i(TAG, "arg=" + arg);
                    }
                }

                Log.i(TAG, "calling runScenarioRunner");
                int rc = runScenarioRunner(args);
                Log.i(TAG, "native exit code=" + rc);

            } catch (Throwable t) {
                Log.e(TAG, "runScenarioRunner failed", t);
            } finally {
                Log.i(TAG, "stopping service");
                stopForeground(STOP_FOREGROUND_REMOVE);
                stopSelf(startId);
            }
        }).start();

        return START_NOT_STICKY;
    }

    private Notification buildNotification() {
        return new Notification.Builder(this, CHANNEL_ID)
            .setContentTitle("Scenario Runner")
            .setContentText("Running...")
            .setSmallIcon(android.R.drawable.stat_notify_sync)
            .build();
    }

    private void createNotificationChannel() {
        NotificationChannel channel =
            new NotificationChannel(CHANNEL_ID, "Scenario Runner", NotificationManager.IMPORTANCE_LOW);
        NotificationManager nm = getSystemService(NotificationManager.class);
        if (nm != null) {
            nm.createNotificationChannel(channel);
        }
    }

    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }
}
