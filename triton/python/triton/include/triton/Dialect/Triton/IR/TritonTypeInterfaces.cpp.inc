/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/// Returns the encoding of the tensor or memory descriptor
mlir::Attribute mlir::TensorOrMemDesc::getEncoding() const {
      return getImpl()->getEncoding(getImpl(), *this);
  }
/// Returns element type
mlir::Type mlir::TensorOrMemDesc::getElementType() const {
      return getImpl()->getElementType(getImpl(), *this);
  }
/// Returns the type shape
llvm::ArrayRef<int64_t> mlir::TensorOrMemDesc::getShape() const {
      return getImpl()->getShape(getImpl(), *this);
  }
/// Returns the tensor or buffer rank
int64_t mlir::TensorOrMemDesc::getRank() const {
      return getImpl()->getRank(getImpl(), *this);
  }
/// Returns the element type bit width
int64_t mlir::TensorOrMemDesc::getElementTypeBitWidth() const {
      return getImpl()->getElementTypeBitWidth(getImpl(), *this);
  }
