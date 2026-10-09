#pragma once
#include "Arms.h"

// Weapon feel and handling, editable in weapon_tuning.json in the config folder (%APPDATA%\MirasEngine).
// The file is read at every level start and restart, so changes show after "Restart Level"; missing keys
// are written back with these defaults. Sections: "sway", "pistol", "shotgun", "hands", "zones",
// "cancelWindows" (see ArmsConfig in Arms.h for the handling values).
struct WeaponTuning {
    // Turning: the gun lags behind the view by turnSway x the angle turned in a frame, at most maxSway,
    // and recovers at swayRecovery per second. Moving multiplies the lag by 1 + moveTurnSway x bob.
    float turnSway = 0.7f;
    float maxSwayDegrees = 11.5f;
    float swayRecovery = 7.0f;
    float moveTurnSway = 0.35f;
    // Steps: the muzzle swings side to side and dips (degrees at walking pace).
    float stepSwayYawDegrees = 1.26f;
    float stepSwayPitchDegrees = 0.92f;
    // Bob: how far the whole gun moves with each step (meters at walking pace).
    float bobSide = 0.012f;
    float bobUp = 0.014f;
    // Sprinting scales the step swing, the bob and the move lag by this (walking = 1).
    float sprintBob = 2.2f;

    ArmsConfig arms;
};

// Reads the file (writing it with the defaults when it doesn't exist or lacks keys).
WeaponTuning loadWeaponTuning();
