[app]
title = Layla
# Layla Personal AI
# Basic Kivy Android project

package.name = layla
package.domain = org.layla
source.dir = .
source.include_exts = py,png,jpg,jpeg,json,kv,atlas
source.exclude_patterns = license, data/screens/three.kv, tests/*, */tests/*
version = 0.1
requirements = python3,kivy,plyer,pillow
orientation = portrait
fullscreen = 0

# Android settings
android.api = 33
android.minapi = 21
android.ndk_api = 21
android.ndk = 25b
android.archs = arm64-v8a
android.accept_sdk_license = True 
android.permissions = INTERNET,RECORD_AUDIO

[buildozer]
log_level = 2
warn_on_root = 1
