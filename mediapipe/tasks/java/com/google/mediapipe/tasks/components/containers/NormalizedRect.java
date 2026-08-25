// Copyright 2026 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.mediapipe.tasks.components.containers;

import com.google.auto.value.AutoValue;

/**
 * A possibly rotated rectangle expressed in normalized image coordinates.
 *
 * <p>The center and dimensions are normalized independently by the image width and height. For
 * example, a square in pixel space generally has different {@link #width()} and {@link #height()}
 * values when the source image is not square. Rotation is clockwise in radians.
 */
@AutoValue
public abstract class NormalizedRect {

  /** Creates a normalized rectangle. */
  public static NormalizedRect create(
      float xCenter, float yCenter, float width, float height, float rotationRadians) {
    return new AutoValue_NormalizedRect(xCenter, yCenter, width, height, rotationRadians);
  }

  /** Creates a normalized rectangle from its protobuf representation. */
  public static NormalizedRect createFromProto(
      com.google.mediapipe.formats.proto.RectProto.NormalizedRect rectProto) {
    return create(
        rectProto.getXCenter(),
        rectProto.getYCenter(),
        rectProto.getWidth(),
        rectProto.getHeight(),
        rectProto.getRotation());
  }

  /** Returns the protobuf representation of this rectangle. */
  public final com.google.mediapipe.formats.proto.RectProto.NormalizedRect toProto() {
    return com.google.mediapipe.formats.proto.RectProto.NormalizedRect.newBuilder()
        .setXCenter(xCenter())
        .setYCenter(yCenter())
        .setWidth(width())
        .setHeight(height())
        .setRotation(rotationRadians())
        .build();
  }

  /** Horizontal center normalized by the source image width. */
  public abstract float xCenter();

  /** Vertical center normalized by the source image height. */
  public abstract float yCenter();

  /** Rectangle width normalized by the source image width. */
  public abstract float width();

  /** Rectangle height normalized by the source image height. */
  public abstract float height();

  /** Clockwise rotation in radians. */
  public abstract float rotationRadians();
}
