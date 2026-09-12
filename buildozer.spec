[app]
title = Layla
# Layla Personal AI
# Basic Kivy Android project

package.name = layla
package.domain = org.layla
source.dir = .
source.include_exts = py,png,jpg,jpeg,json
version = 0.1
requirements = python3,kivy
orientation = portrait
fullscreen = 0

# Android settings
android.api = 35
android.minapi = 21
android.archs = arm64-v8a, armeabi-v7a

[buildozer]
log_level = 2
warn_on_root = 1
