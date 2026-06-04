/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
class TensorOrMemDesc;
namespace detail {
struct TensorOrMemDescInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    mlir::Attribute (*getEncoding)(const Concept *impl, ::mlir::Type );
    mlir::Type (*getElementType)(const Concept *impl, ::mlir::Type );
    llvm::ArrayRef<int64_t> (*getShape)(const Concept *impl, ::mlir::Type );
    int64_t (*getRank)(const Concept *impl, ::mlir::Type );
    int64_t (*getElementTypeBitWidth)(const Concept *impl, ::mlir::Type );
  };
  template<typename ConcreteType>
  class Model : public Concept {
  public:
    using Interface = ::mlir::TensorOrMemDesc;
    Model() : Concept{getEncoding, getElementType, getShape, getRank, getElementTypeBitWidth} {}

    static inline mlir::Attribute getEncoding(const Concept *impl, ::mlir::Type tablegen_opaque_val);
    static inline mlir::Type getElementType(const Concept *impl, ::mlir::Type tablegen_opaque_val);
    static inline llvm::ArrayRef<int64_t> getShape(const Concept *impl, ::mlir::Type tablegen_opaque_val);
    static inline int64_t getRank(const Concept *impl, ::mlir::Type tablegen_opaque_val);
    static inline int64_t getElementTypeBitWidth(const Concept *impl, ::mlir::Type tablegen_opaque_val);
  };
  template<typename ConcreteType>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::TensorOrMemDesc;
    FallbackModel() : Concept{getEncoding, getElementType, getShape, getRank, getElementTypeBitWidth} {}

    static inline mlir::Attribute getEncoding(const Concept *impl, ::mlir::Type tablegen_opaque_val);
    static inline mlir::Type getElementType(const Concept *impl, ::mlir::Type tablegen_opaque_val);
    static inline llvm::ArrayRef<int64_t> getShape(const Concept *impl, ::mlir::Type tablegen_opaque_val);
    static inline int64_t getRank(const Concept *impl, ::mlir::Type tablegen_opaque_val);
    static inline int64_t getElementTypeBitWidth(const Concept *impl, ::mlir::Type tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteType>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteType;
  };
};
template <typename ConcreteType>
struct TensorOrMemDescTrait;

} // namespace detail
class TensorOrMemDesc : public ::mlir::TypeInterface<TensorOrMemDesc, detail::TensorOrMemDescInterfaceTraits> {
public:
  using ::mlir::TypeInterface<TensorOrMemDesc, detail::TensorOrMemDescInterfaceTraits>::TypeInterface;
  template <typename ConcreteType>
  struct Trait : public detail::TensorOrMemDescTrait<ConcreteType> {};
  /// Returns the encoding of the tensor or memory descriptor
  mlir::Attribute getEncoding() const;
  /// Returns element type
  mlir::Type getElementType() const;
  /// Returns the type shape
  llvm::ArrayRef<int64_t> getShape() const;
  /// Returns the tensor or buffer rank
  int64_t getRank() const;
  /// Returns the element type bit width
  int64_t getElementTypeBitWidth() const;
};
namespace detail {
  template <typename ConcreteType>
  struct TensorOrMemDescTrait : public ::mlir::TypeInterface<TensorOrMemDesc, detail::TensorOrMemDescInterfaceTraits>::Trait<ConcreteType> {
  };
}// namespace detail
} // namespace mlir
namespace mlir {
template<typename ConcreteType>
mlir::Attribute detail::TensorOrMemDescInterfaceTraits::Model<ConcreteType>::getEncoding(const Concept *impl, ::mlir::Type tablegen_opaque_val) {
  return (::llvm::cast<ConcreteType>(tablegen_opaque_val)).getEncoding();
}
template<typename ConcreteType>
mlir::Type detail::TensorOrMemDescInterfaceTraits::Model<ConcreteType>::getElementType(const Concept *impl, ::mlir::Type tablegen_opaque_val) {
  return (::llvm::cast<ConcreteType>(tablegen_opaque_val)).getElementType();
}
template<typename ConcreteType>
llvm::ArrayRef<int64_t> detail::TensorOrMemDescInterfaceTraits::Model<ConcreteType>::getShape(const Concept *impl, ::mlir::Type tablegen_opaque_val) {
  return (::llvm::cast<ConcreteType>(tablegen_opaque_val)).getShape();
}
template<typename ConcreteType>
int64_t detail::TensorOrMemDescInterfaceTraits::Model<ConcreteType>::getRank(const Concept *impl, ::mlir::Type tablegen_opaque_val) {
  return (::llvm::cast<ConcreteType>(tablegen_opaque_val)).getRank();
}
template<typename ConcreteType>
int64_t detail::TensorOrMemDescInterfaceTraits::Model<ConcreteType>::getElementTypeBitWidth(const Concept *impl, ::mlir::Type tablegen_opaque_val) {
  return (::llvm::cast<ConcreteType>(tablegen_opaque_val)).getElementTypeBitWidth();
}
template<typename ConcreteType>
mlir::Attribute detail::TensorOrMemDescInterfaceTraits::FallbackModel<ConcreteType>::getEncoding(const Concept *impl, ::mlir::Type tablegen_opaque_val) {
  return static_cast<const ConcreteType *>(impl)->getEncoding(tablegen_opaque_val);
}
template<typename ConcreteType>
mlir::Type detail::TensorOrMemDescInterfaceTraits::FallbackModel<ConcreteType>::getElementType(const Concept *impl, ::mlir::Type tablegen_opaque_val) {
  return static_cast<const ConcreteType *>(impl)->getElementType(tablegen_opaque_val);
}
template<typename ConcreteType>
llvm::ArrayRef<int64_t> detail::TensorOrMemDescInterfaceTraits::FallbackModel<ConcreteType>::getShape(const Concept *impl, ::mlir::Type tablegen_opaque_val) {
  return static_cast<const ConcreteType *>(impl)->getShape(tablegen_opaque_val);
}
template<typename ConcreteType>
int64_t detail::TensorOrMemDescInterfaceTraits::FallbackModel<ConcreteType>::getRank(const Concept *impl, ::mlir::Type tablegen_opaque_val) {
  return static_cast<const ConcreteType *>(impl)->getRank(tablegen_opaque_val);
}
template<typename ConcreteType>
int64_t detail::TensorOrMemDescInterfaceTraits::FallbackModel<ConcreteType>::getElementTypeBitWidth(const Concept *impl, ::mlir::Type tablegen_opaque_val) {
  return static_cast<const ConcreteType *>(impl)->getElementTypeBitWidth(tablegen_opaque_val);
}
} // namespace mlir
