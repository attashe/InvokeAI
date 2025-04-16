from typing import Literal, Optional, List

from invokeai.app.invocations.baseinvocation import (
    BaseModel,
    BaseInvocation,
    BaseInvocationOutput,
    Classification,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.fields import (
    InputField,
    OutputField,
    FluxUnoReferenceField
)
from invokeai.app.invocations.primitives import ImageField
from invokeai.app.services.shared.invocation_context import InvocationContext

@invocation_output("flux_uno_output")
class FluxUnoOutput(BaseInvocationOutput):
    """The conditioning output of a FLUX Redux invocation."""

    uno_refs: FluxUnoReferenceField = OutputField(
        description="Reference images container", title="Reference images"
    )

# TODO(attashe): adjust tags and category
@invocation(
    "flux_uno",
    title="FLUX UNO",
    tags=["ip_adapter", "control"],
    category="ip_adapter",
    version="2.1.0",
    classification=Classification.Beta,
)
class FluxReduxInvocation(BaseInvocation):
    """Runs a FLUX Redux model to generate a conditioning tensor."""

    image: ImageField = InputField(description="The FLUX Redux image prompt.")
    image2: Optional[ImageField] = InputField(default=None, description="2nd reference")
    image3: Optional[ImageField] = InputField(default=None, description="3rd reference")
    image4: Optional[ImageField] = InputField(default=None, description="4th reference")

    def invoke(self, context: InvocationContext) -> FluxUnoOutput:

        images: list[str] = []
        for image in [self.image, self.image2, self.image3, self.image4]:
            if image is not None:
                image_pil = context.images.get_pil(self.image.image_name)
                images.append(context.images.save(image=image_pil).image_name)
        
        return FluxUnoOutput(
            uno_refs=FluxUnoReferenceField(
                image_names=images)
        )
