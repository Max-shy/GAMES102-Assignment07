// This file is generated by Ubpa::USRefl::AutoRefl

#pragma once

#include <USRefl/USRefl.h>

template<>
struct Ubpa::USRefl::TypeInfo<DenoiseData> :
    TypeInfoBase<DenoiseData>
{
#ifdef UBPA_USREFL_NOT_USE_NAMEOF
    static constexpr char name[12] = "DenoiseData";
#endif
    static constexpr AttrList attrs = {};
    static constexpr FieldList fields = {
        Field {TSTR("randomScale"), &Type::randomScale, AttrList {
            Attr {TSTR(UMeta::initializer), []()->float{ return 1.f; }},
            Attr {TSTR(UInspector::min_value), 0.f},
            Attr {TSTR(UInspector::tooltip), "random scale"},
        }},
        Field {TSTR("mesh"), &Type::mesh},
        Field {TSTR("heMesh"), &Type::heMesh, AttrList {
            Attr {TSTR(UMeta::initializer), []()->std::shared_ptr<HEMeshX>{ return { std::make_shared<HEMeshX>() }; }},
            Attr {TSTR(UInspector::hide)},
        }},
        Field {TSTR("copy"), &Type::copy, AttrList {
            Attr {TSTR(UInspector::hide)},
        }},
    };
};

